# Based on: https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/gpt/utils/gpt_decoder.py # noqa: E501
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Union
import os

import numpy as np

from . import comm
from . import profiler
from .gpt import GptInitModelParameters

from nebullvm.optional_modules.torch import torch

PathLike = Union[str, Path]


def to_numpy_dtype(maybe_str_dtype: Union[str, np.dtype]):
    assert isinstance(maybe_str_dtype, (str, np.dtype))
    if isinstance(maybe_str_dtype, str):
        try:
            dtype = {
                "fp16": np.float16,
                "float16": np.float16,
                "fp32": np.float32,
                "float32": np.float32,
            }[maybe_str_dtype]
        except KeyError:
            raise ValueError(
                f"Cannot convert to numpy data type, got {maybe_str_dtype}"
            )
    else:
        dtype = maybe_str_dtype
    return dtype


def to_torch_dtype(maybe_str_dtype: Union[str, torch.dtype]):

    if isinstance(maybe_str_dtype, torch.dtype):
        dtype = maybe_str_dtype
    else:
        try:
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[maybe_str_dtype]
        except KeyError:
            raise ValueError(
                f"Cannot convert to torch data type, got {maybe_str_dtype}"
            )
    return dtype


def load_weight_from_bin(
    checkpoint_path: PathLike,
    shape: List[int],
    weight_dtype: Union[str, np.dtype],
):
    """Load a weight from a bin file.

    # Args.
        checkpoint_path: str or Path,
            a checkpoint file path of an FT's layer weight.
        shape: list of int, the shape of weight tensor.
        weight_dtype: str or np.dtype, the data type of the stored weight.
    """
    weight_dtype = to_numpy_dtype(weight_dtype)
    return torch.from_numpy(np.fromfile(checkpoint_path, dtype=weight_dtype))


LayernormType = Literal["pre_layernorm", "post_layernorm"]


class GptLayerWeights:
    def __init__(
        self,
        num_heads: int,
        size_per_head: int,
        inter_size: int,
        num_layers: int,
        tensor_para_size: int = 1,
        pipeline_para_size: int = 1,
        has_adapters: bool = False,
        adapter_inter_size: int = 0,
        int8_mode: int = 0,
    ):

        assert num_heads % tensor_para_size == 0, (
            f"num_heads ({num_heads}) is not multiple of "
            "tensor para size ({tensor_para_size})"
        )

        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_units = num_heads * size_per_head
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.tensor_para_rank = comm.get_tensor_para_rank()
        self.pipeline_para_size = pipeline_para_size
        self.pipeline_para_rank = comm.get_pipeline_para_rank()

        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        self.local_num_layers = num_layers // pipeline_para_size
        self.local_num_heads = num_heads // tensor_para_size
        self.local_hidden_units = self.local_num_heads * size_per_head
        self.local_inter_size = inter_size // tensor_para_size
        self.local_adapter_inter_size = (
            self.adapter_inter_size // tensor_para_size
        )

        self.weight_transpose_calibrate_quantize = None
        assert int8_mode in [0, 1], "Invalid int8 mode for GPT. Must be 0 or 1"
        self.int8_mode = int8_mode
        if self.int8_mode == 1:
            quant = (
                torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix  # noqa: E501
            )
            self.weight_transpose_calibrate_quantize = lambda x: quant(
                x, torch.int8
            )

        self.weights = None
        self.int8_weights = None
        self.int8_scales = None

        self.expected_weight_shapes = list()

        # pylint:disable=line-too-long
        # Transformer blocks
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # input layernorm weight
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # input layernorm bias
        self.expected_weight_shapes.extend(
            [(self.hidden_units, self.local_hidden_units * 3)]
            * self.local_num_layers
        )  # attention qkv weight
        self.expected_weight_shapes.extend(
            [(self.local_hidden_units * 3,)] * self.local_num_layers
        )  # attention qkv bias
        self.expected_weight_shapes.extend(
            [(self.local_hidden_units, self.hidden_units)]
            * self.local_num_layers
        )  # attention dense weight
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # attention dense bias
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # post attention layernorm weight
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # post attention layernorm bias
        self.expected_weight_shapes.extend(
            [(self.hidden_units, self.local_inter_size)]
            * self.local_num_layers
        )  # ffn_kernel1
        self.expected_weight_shapes.extend(
            [(self.local_inter_size,)] * self.local_num_layers
        )  # ffn_bias1
        self.expected_weight_shapes.extend(
            [(self.local_inter_size, self.hidden_units)]
            * self.local_num_layers
        )  # ffn_kernel2
        self.expected_weight_shapes.extend(
            [(self.hidden_units,)] * self.local_num_layers
        )  # ffn_bias2

        # Adapters
        if self.has_adapters:
            self.expected_weight_shapes.extend(
                [(self.hidden_units, self.local_adapter_inter_size)]
                * self.local_num_layers
            )  # adaptor1_kernel1
            self.expected_weight_shapes.extend(
                [(self.local_adapter_inter_size,)] * self.local_num_layers
            )  # adaptor1_bias1
            self.expected_weight_shapes.extend(
                [(self.local_adapter_inter_size, self.hidden_units)]
                * self.local_num_layers
            )  # adaptor1_kernel2
            self.expected_weight_shapes.extend(
                [(self.hidden_units,)] * self.local_num_layers
            )  # adaptor1_bias2
            self.expected_weight_shapes.extend(
                [(self.hidden_units, self.local_adapter_inter_size)]
                * self.local_num_layers
            )  # adaptor2_kernel1
            self.expected_weight_shapes.extend(
                [(self.local_adapter_inter_size,)] * self.local_num_layers
            )  # adaptor2_bias1
            self.expected_weight_shapes.extend(
                [(self.local_adapter_inter_size, self.hidden_units)]
                * self.local_num_layers
            )  # adaptor2_kernel2
            self.expected_weight_shapes.extend(
                [(self.hidden_units,)] * self.local_num_layers
            )  # adaptor2_bias2
        # pylint:enable=line-too-long

    @classmethod
    def from_config(cls, config: GptInitModelParameters):
        return cls(
            num_heads=config.head_num,
            size_per_head=config.size_per_head,
            inter_size=4 * config.head_num * config.size_per_head,
            num_layers=config.layer_num,
            tensor_para_size=config.tensor_para_size,
            pipeline_para_size=config.pipeline_para_size,
            has_adapters=config.has_adapters,
            adapter_inter_size=config.adapter_inter_size,
            int8_mode=config.int8_mode,
        )

    @property
    def dtype(self):
        return self.weights[0].dtype

    @property
    def device(self):
        return self.weights[0].device

    def _map(self, func):
        for i in range(len(self.weights)):
            if isinstance(self.weights[i], list):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = func(self.weights[i][j])
            else:
                self.weights[i] = func(self.weights[i])

    def _map_int8(self, func):
        for i in range(len(self.int8_weights)):
            if isinstance(self.int8_weights[i], list):
                for j in range(len(self.int8_weights[i])):
                    self.int8_weights[i][j] = func(self.int8_weights[i][j])

            else:
                self.int8_weights[i] = func(self.int8_weights[i])
        for i in range(len(self.int8_scales)):
            if isinstance(self.int8_scales[i], list):
                for j in range(len(self.int8_scales[i])):
                    self.int8_scales[i][j] = func(self.int8_scales[i][j])
            else:
                self.int8_scales[i] = func(self.int8_scales[i])

    def float(self):
        if self.dtype == torch.float32:
            return
        self._map(lambda x: x.float())

    def half(self):
        if self.dtype == torch.float16:
            return
        self._map(lambda x: x.half())
        if self.int8_mode == 1:
            self._map_int8(lambda w: w.half())

    def bfloat16(self):
        if self.dtype == torch.bfloat16:
            return
        self._map(lambda x: x.bfloat16())
        if self.int8_mode == 1:
            self._map_int8(lambda w: w.bfloat16())

    def cuda(self, device=None):
        self._map(lambda x: x.cuda(device))
        if self.int8_mode == 1:
            self._map_int8(lambda x: x.cuda(device))

    def to(self, device=None):
        self._map(lambda x: x.to(device))
        if self.int8_mode == 1:
            self._map_int8(lambda x: x.to(device))

    def is_valid_pp_group(self, layer, pp_rank):
        return layer // self.layers_per_device == pp_rank

    def load(
        self,
        checkpoint_path: PathLike,
        compute_dtype: torch.dtype,
        weight_dtype: Optional[Union[str, np.dtype]] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ):
        """Load checkpoint weights.

        # Args.
            checkpoint_path: str or Path,
                a checkpoint directory where FT checkpoint files locate.
            weight_dtype: str or np.dtype, the data type of stored weights.
        """

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Could not find checkpoint {str(checkpoint_path)}"
            )

        weight_dtype = to_numpy_dtype(weight_dtype)
        print(
            f"Load weights from {str(checkpoint_path)} (data type: {weight_dtype}"  # noqa: E501
        )

        self.weights = list()
        self.int8_weights = list()
        self.int8_scales = list()
        torch.cuda.empty_cache()

        def _load_from_file(fname):
            quant_sub_names = [
                "attention.query_key_value.weight",
                "attention.dense.weight",
                "dense_h_to_4h.weight",
                "dense_4h_to_h.weight",
            ]
            _weight = torch.from_numpy(
                np.fromfile(checkpoint_path / fname, dtype=weight_dtype)
            )
            _weight = _weight.to(compute_dtype)
            weight_index = len(self.weights)
            expected_shape = self.expected_weight_shapes[weight_index]

            try:
                if _weight.nelement() > 0:
                    _weight = _weight.reshape(expected_shape)
            except:  # noqa: E722
                raise ValueError(
                    f"num_heads, size_per_head, vocab_size, and max_seq_len must be the same "  # noqa: E501
                    f"as the ones during training (weight: {fname} expected shape: {expected_shape}, "  # noqa: E501
                    f"got shape: {_weight.shape})."
                )

            should_quantize = any(
                sub_name in fname for sub_name in quant_sub_names
            )
            if self.int8_mode != 0 and should_quantize:
                calibrate = self.weight_transpose_calibrate_quantize
                int8_weight, int8_scales = calibrate(_weight)

                # int8 weights should appear in same order as FP weights.
                # Move to device and add to the int8 list.
                dummy_weight = torch.empty(0, dtype=compute_dtype)
                if device is not None:
                    int8_weight = int8_weight.to(device)
                    int8_scales = int8_scales.to(device)
                    dummy_weight = dummy_weight.to(device)

                self.int8_weights.append(int8_weight)
                self.int8_scales.append(int8_scales)
                self.weights.append(dummy_weight)
            else:
                if device is not None:
                    _weight = _weight.to(device)
                self.weights.append(_weight)

        # Load
        # pylint:disable=line-too-long
        layer_offset = self.local_num_layers * self.pipeline_para_rank
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.input_layernorm.weight.bin"
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.input_layernorm.bias.bin"
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.attention.query_key_value.weight.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.attention.query_key_value.bias.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.attention.dense.weight.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.attention.dense.bias.bin"
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.post_attention_layernorm.weight.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.post_attention_layernorm.bias.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
            )
            for i in range(self.local_num_layers)
        ]
        [
            _load_from_file(
                f"model.layers.{layer_offset + i}.mlp.dense_4h_to_h.bias.bin"
            )
            for i in range(self.local_num_layers)
        ]

        if self.has_adapters:
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_attention_adapter.dense_h_to_4h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_attention_adapter.dense_h_to_4h.bias.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_attention_adapter.dense_4h_to_h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_attention_adapter.dense_4h_to_h.bias.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_ffn_adapter.dense_h_to_4h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_ffn_adapter.dense_h_to_4h.bias.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_ffn_adapter.dense_4h_to_h.weight.{self.tensor_para_rank}.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]
            [
                _load_from_file(
                    f"model.layers.{layer_offset + i}.after_ffn_adapter.dense_4h_to_h.bias.bin"  # noqa: E501
                )
                for i in range(self.local_num_layers)
            ]

        assert len(self.weights) == len(
            self.expected_weight_shapes
        ), "Incorrect number of weights loaded"


class FtModuleBase:
    def __init__(self):
        self.weight = None

    @classmethod
    @abstractmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _initialize_model(self, force_init=False):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_weight(self, weight: GptLayerWeights):
        old_weight_dtype = (
            self.weight.dtype if self.weight is not None else None
        )
        self.weight = weight
        if old_weight_dtype is None or old_weight_dtype != self.weight.dtype:
            self._initialize_model(force_init=True)

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device

    def cuda(self, device=None):
        assert torch.cuda.is_available()
        self.weight.cuda(device)
        return self

    def to(self, device=None):
        self.weight.to(device)
        return self

    def float(self):
        self.weight.float()
        self._initialize_model(force_init=True)
        return self

    def half(self):
        self.weight.half()
        self._initialize_model(force_init=True)
        return self

    def bfloat16(self):
        self.weight.bfloat16()
        self._initialize_model(force_init=True)
        return self


class GptContextDecoder(FtModuleBase):
    def __init__(
        self,
        num_heads: int,
        size_per_head: int,
        inter_size: int,
        num_layers: int,
        tensor_para_size: int = 1,
        pipeline_para_size: int = 1,
        remove_padding: bool = True,
        shared_contexts_ratio: float = 1.0,
        layernorm_eps: float = 1e-6,
        layernorm_type: LayernormType = "pre_layernorm",
        activation_type: str = "gelu",
        has_adapters: bool = False,
        adapter_inter_size: int = 0,
        int8_mode: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_size = self.num_heads * self.size_per_head
        self.inter_size = inter_size
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size

        self.remove_padding = remove_padding
        self.shared_contexts_ratio = shared_contexts_ratio

        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        assert int8_mode in [0, 1]
        self.int8_mode = int8_mode

        self.ft_op = None
        self.weight = None

    def __repr__(self):
        args_dict = dict(
            num_heads=self.num_heads,
            size_per_head=self.size_per_head,
            hidden_size=self.hidden_size,
            inter_size=self.inter_size,
            num_layers=self.num_layers,
            tensor_para_size=self.tensor_para_size,
            pipeline_para_size=self.pipeline_para_size,
            remove_padding=self.remove_padding,
            shared_contexts_ratio=self.shared_contexts_ratio,
            layernorm_eps=self.layernorm_eps,
            layernorm_type=self.layernorm_type,
            activation_type=self.activation_type,
            has_adapters=self.has_adapters,
            adapter_inter_size=self.adapter_inter_size,
            int8_mode=self.int8_mode,
        )
        args_str = ",\n    ".join([f"{k}: {v}" for k, v in args_dict.items()])
        return f"{self.__class__.__name__}[\n{    args_str}\n]"

    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        return cls(
            num_heads=config.head_num,
            size_per_head=config.size_per_head,
            inter_size=4 * config.head_num * config.size_per_head,
            num_layers=config.layer_num,
            tensor_para_size=config.tensor_para_size,
            pipeline_para_size=config.pipeline_para_size,
            remove_padding=kwargs.get("remove_padding", True),
            shared_contexts_ratio=kwargs.get("shared_contexts_ratio", 1.0),
            layernorm_eps=config.layernorm_eps,
            layernorm_type=config.layernorm_type,
            activation_type=config.activation_type,
            has_adapters=config.has_adapters,
            adapter_inter_size=config.adapter_inter_size,
            int8_mode=config.int8_mode,
        )

    def _initialize_model(self, force_init=False):
        if self.weight is None:
            self.weight = GptLayerWeights(
                num_heads=self.num_heads,
                size_per_head=self.size_per_head,
                inter_size=self.inter_size,
                num_layers=self.num_layers,
                tensor_para_size=self.tensor_para_size,
                pipeline_para_size=self.pipeline_para_size,
                has_adapters=self.has_adapters,
                adapter_inter_size=self.adapter_inter_size,
                int8_mode=self.int8_mode,
            )
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op

        self.ft_op = (
            torch.classes.FasterTransformer.ParallelGptContextDecoderOp(
                self.num_heads,
                self.size_per_head,
                self.inter_size,
                self.num_layers,
                self.tensor_para_size,
                self.pipeline_para_size,
                self.layernorm_eps,
                self.layernorm_type,
                self.activation_type,
                self.has_adapters,
                self.adapter_inter_size,
                self.int8_mode,
                self.weight.weights,
                self.weight.int8_weights,
                self.weight.int8_scales,
                self.remove_padding,
            )
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        input_lengths: torch.IntTensor,
        memory_length: Optional[int] = None,
        compact_index: Optional[torch.IntTensor] = None,
        batch_to_compact_index: Optional[torch.IntTensor] = None,
        linear_bias_slopes: Optional[torch.Tensor] = None,
    ):
        """

        # Args.
            input_embeds: Tensor, (batch * beam, max_input_length, hidden_dim),
                input hidden states.
            attention_mask: Tensor, (batch * beam, max_input_length, max_input_length),
                input attention mask.
            input_lengths: (batch * beam,), input sequence lengths.
            memory_length: int, the length of memory to keep key/cache values.
            compact_index: IntTensor, (compact_batch_size,)
                The index of input sequences of a compact batch. If None, the FT op
                doesn't apply the shared context feature and as result the inference
                time may increase.
            batch_to_compact_index: IntTensor, (batch * beam,)
                The index map from the original input batch to the compact batch.
                This must be provided if compact_index is not None.
            linear_bias_slopes: (num_heads,)
                The slope per head of linear attention bias - ALiBi. If None, a base
                self attention will be performed.
        # Returns
            hidden_states: Tensor, (batch * beam, max_input_length, hidden_dim),
                decoder outputs.
            key_cache: Tensor, (num_layers, batch * beam, local_num_heads, size_per_head / x, memory_length, x), # noqa: E501
                key cache of attention of inputs.
                x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length # noqa: E501
            value_cache: Tensor, (num_layers, batch * beam, local_num_heads, memory_length, hidden_dim) # noqa: E501
                value cache of attention
            last_token_hidden_states: Tensor, (batch * beam, hidden_dim)
                hidden states of the last input token.
        """
        self._initialize_model()
        # outputs: output hidden states
        (
            decoder_ouptut,
            key_cache,
            value_cache,
            last_token_hidden_states,
        ) = self.ft_op.forward(
            input_embeds,
            attention_mask,
            input_lengths,
            memory_length,
            compact_index,
            batch_to_compact_index,
            linear_bias_slopes,
        )
        return decoder_ouptut, key_cache, value_cache, last_token_hidden_states


class GptDecoder(FtModuleBase):
    def __init__(
        self,
        num_heads: int,
        size_per_head: int,
        inter_size: int,
        num_layers: int,
        tensor_para_size: int = 1,
        pipeline_para_size: int = 1,
        layernorm_eps: float = 1e-6,
        layernorm_type: LayernormType = "pre_layernorm",
        activation_type: str = "gelu",
        has_adapters: bool = False,
        adapter_inter_size: int = 0,
        int8_mode: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_size = self.num_heads * self.size_per_head
        self.inter_size = inter_size
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size

        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        self.int8_mode = int8_mode

        self.ft_op = None
        self.weight = None

    def __repr__(self):
        args_dict = dict(
            num_heads=self.num_heads,
            size_per_head=self.size_per_head,
            hidden_size=self.hidden_size,
            inter_size=self.inter_size,
            num_layers=self.num_layers,
            tensor_para_size=self.tensor_para_size,
            pipeline_para_size=self.pipeline_para_size,
            layernorm_eps=self.layernorm_eps,
            layernorm_type=self.layernorm_type,
            activation_type=self.activation_type,
            has_adapters=self.has_adapters,
            adapter_inter_size=self.adapter_inter_size,
            int8_mode=self.int8_mode,
        )
        args_str = ",\n    ".join(
            [f"{k}: {v}" for k, v in args_dict.items()]
        )  # noqa: E501
        return f"{self.__class__.__name__}[\n    {args_str}\n]"

    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        hidden_dim = config.head_num * config.size_per_head
        return cls(
            num_heads=config.head_num,
            size_per_head=config.size_per_head,
            inter_size=4 * hidden_dim,
            num_layers=config.layer_num,
            tensor_para_size=config.tensor_para_size,
            pipeline_para_size=config.pipeline_para_size,
            layernorm_eps=config.layernorm_eps,
            layernorm_type=config.layernorm_type,
            activation_type=config.activation_type,
            has_adapters=config.has_adapters,
            adapter_inter_size=config.adapter_inter_size,
            int8_mode=config.int8_mode,
        )

    def _initialize_model(self, force_init=False):
        if self.weight is None:
            self.weight = GptLayerWeights(
                num_heads=self.num_heads,
                size_per_head=self.size_per_head,
                inter_size=self.inter_size,
                num_layers=self.num_layers,
                tensor_para_size=self.tensor_para_size,
                pipeline_para_size=self.pipeline_para_size,
                has_adapters=self.has_adapters,
                adapter_inter_size=self.adapter_inter_size,
                int8_mode=self.int8_mode,
            )
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op
        self.ft_op = torch.classes.FasterTransformer.ParallelGptDecoderOp(
            self.num_heads,
            self.size_per_head,
            self.inter_size,
            self.num_layers,
            self.tensor_para_size,
            self.pipeline_para_size,
            self.layernorm_eps,
            self.layernorm_type,
            self.activation_type,
            self.has_adapters,
            self.adapter_inter_size,
            self.weight.int8_mode,
            self.weight.weights,
            self.weight.int8_weights,
            self.weight.int8_scales,
        )

    def forward(
        self,
        max_input_length: int,
        step: int,
        ite: int,
        input_embeds: torch.Tensor,
        sequence_lengths: torch.IntTensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        finished: torch.BoolTensor,
        total_padding_tokens: torch.IntTensor,
        masked_tokens: torch.BoolTensor,
        cache_indirection: Optional[torch.IntTensor] = None,
        linear_bias_slopes: Optional[torch.Tensor] = None,
    ):
        """

        # Args.
            max_input_length: int, maximum input context length.
            step: int, the current step index.
            ite: int, local batch iteration.
            input_embeds: Tensor, (local_batch * beam, hidden_dim),
                input hidden state to decoder.
            sequence_lengths: IntTensor, (local_batch * beam,),
                the current sequence lengths.
            key_cache: Tensor, key cache buffer.
            value_cache: Tensor, value cache buffer.
            finished: BoolTensor, (local_batch * beam,),
                whether to finish sentence generation.
            total_padding_tokens IntTensor, (local_batch * beam,),
                the number of padded tokens.
            masked_tokens: BoolTensor, (local_batch * beam, memory_length),
                a mask tensor that indicates padded tokens.
            cache_indirection: IntTensor, (local_batch * beam,),
                cache of beam positions if needed if beam > 1.
            linear_bias_slopes Tensor, (num_heads,)
                slopes head of linear position bias (ALiBi) (optional).
        # Returns
            IntTensor, (batch * beam,) output token ids.
        """

        self._initialize_model()

        outputs = self.ft_op.forward(
            max_input_length,
            step,
            ite,
            input_embeds,
            sequence_lengths,
            finished,
            total_padding_tokens,
            masked_tokens,
            key_cache,
            value_cache,
            cache_indirection,
            linear_bias_slopes,
        )
        return outputs[0]


class Gpt:
    def __init__(
        self,
        num_heads: int,
        size_per_head: int,
        num_layers: int,
        vocab_size: int,
        start_id: int,
        end_id: int,
        lib_path: PathLike,
        tensor_para_size: int = 1,
        pipeline_para_size: int = 1,
        remove_padding: bool = True,
        shared_contexts_ratio: float = 1.0,
        layernorm_eps: float = 1e-6,
        layernorm_type: LayernormType = "pre_layernorm",
        activation_type: str = "gelu",
        has_positional_encoding: bool = True,
        max_seq_len: int = 0,
        has_pre_decoder_layernorm: bool = False,
        has_post_decoder_layernorm: bool = True,
        has_adapters: bool = False,
        adapter_inter_size: int = 0,
        int8_mode: int = 0,
        inference_data_type: Optional[str] = None,
        weights_data_type: str = "fp32",
        use_fp32_to_compute_logit: bool = False,
        **kwargs,
    ):
        super().__init__()

        inference_data_type = inference_data_type or weights_data_type

        self.config = GptInitModelParameters(
            head_num=num_heads,
            size_per_head=size_per_head,
            layer_num=num_layers,
            max_seq_len=max_seq_len,
            tensor_para_size=tensor_para_size,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            pipeline_para_size=pipeline_para_size,
            data_type=inference_data_type,
            weights_data_type=weights_data_type,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            activation_type=activation_type,
            has_positional_encoding=has_positional_encoding,
            has_pre_decoder_layernorm=has_pre_decoder_layernorm,
            has_post_decoder_layernorm=has_post_decoder_layernorm,
            has_adapters=has_adapters,
            adapter_inter_size=adapter_inter_size,
            int8_mode=int8_mode,
            sparse=kwargs.get("sparse", False),
        )
        self.use_fp32_to_compute_logit = use_fp32_to_compute_logit

        self.weight = None
        self.shared_contexts_ratio = shared_contexts_ratio

        torch.classes.load_library(os.path.abspath(lib_path))

        # Embeddings to encode or decode tokens.
        hidden_dim = num_heads * size_per_head

        # Pad vocab size for FT.
        local_vocab_size = math.ceil(
            self.config.vocab_size / self.config.tensor_para_size
        )
        if self.config.data_type == "fp16":
            local_vocab_size = math.ceil(local_vocab_size / 8) * 8
        self.vocab_size_padded = (
            local_vocab_size * self.config.tensor_para_size
        )
        self.vocab_size = self.config.vocab_size

        self.decode_op = torch.classes.FasterTransformer.DynamicDecodeOp(
            self.vocab_size,
            self.vocab_size_padded,
            self.config.tensor_para_size,
            self.config.pipeline_para_size,
            torch.float,
        )

        self._parameters = {}

        def register_param(name, p):
            self._parameters[name] = p
            setattr(self, name, p)

        register_param(
            "context_decoder",
            GptContextDecoder.from_config(
                self.config,
                remove_padding=remove_padding,
                shared_contexts_ratio=shared_contexts_ratio,
                **kwargs,
            ),
        )
        register_param(
            "decoder", GptDecoder.from_config(self.config, **kwargs)
        )

        compute_dtype = to_torch_dtype(inference_data_type)

        if comm.is_pipeline_group_first():
            register_param(
                "word_embedding",
                torch.nn.Embedding(
                    self.vocab_size_padded, hidden_dim, dtype=compute_dtype
                ),
            )
            self._mask_padded_vocab_weights(self.word_embedding.weight)
            if self.config.has_positional_encoding:
                register_param(
                    "position_encoding",
                    torch.nn.Embedding(
                        self.config.max_seq_len,
                        hidden_dim,
                        dtype=compute_dtype,
                    ),
                )
            else:
                self.position_encoding = None
            if self.config.has_pre_decoder_layernorm:
                register_param(
                    "pre_decoder_layernorm",
                    torch.nn.LayerNorm(
                        hidden_dim, eps=layernorm_eps, dtype=compute_dtype
                    ),
                )
            else:
                self.pre_decoder_layernorm = None

        if comm.is_pipeline_group_last():
            if has_post_decoder_layernorm:
                register_param(
                    "post_decoder_layernorm",
                    torch.nn.LayerNorm(
                        hidden_dim, eps=layernorm_eps, dtype=compute_dtype
                    ),
                )
            else:
                self.post_decoder_layernorm = None
            self.lm_head_ctype = (
                compute_dtype
                if not self.use_fp32_to_compute_logit
                else torch.float32
            )
            register_param(
                "lm_head",
                torch.nn.Linear(
                    hidden_dim,
                    self.vocab_size_padded,
                    bias=False,
                    dtype=self.lm_head_ctype,
                ),
            )
            self._mask_padded_vocab_weights(self.lm_head.weight)

    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        return cls(
            num_heads=config.head_num,
            size_per_head=config.size_per_head,
            num_layers=config.layer_num,
            max_seq_len=config.max_seq_len,
            tensor_para_size=config.tensor_para_size,
            vocab_size=config.vocab_size,
            start_id=config.start_id,
            end_id=config.end_id,
            pipeline_para_size=config.pipeline_para_size,
            inference_data_type=config.data_type,
            weights_data_type=config.weights_data_type,
            layernorm_eps=config.layernorm_eps,
            layernorm_type=config.layernorm_type,
            activation_type=config.activation_type,
            has_positional_encoding=config.has_positional_encoding,
            has_pre_decoder_layernorm=config.has_pre_decoder_layernorm,
            has_post_decoder_layernorm=config.has_post_decoder_layernorm,
            has_adapters=config.has_adapters,
            adapter_inter_size=config.adapter_inter_size,
            int8_mode=config.int8_mode,
            **kwargs,
        )

    def load(
        self,
        checkpoint_path: PathLike,
        inference_data_type: Optional[Union[str, torch.dtype]] = None,
        config: Optional[GptInitModelParameters] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ):

        checkpoint_path = Path(checkpoint_path)
        device = device or comm.get_device()
        config = config or self.config

        compute_dtype = to_torch_dtype(inference_data_type or self.dtype)

        self.weight = GptLayerWeights.from_config(config)
        self.weight.load(
            checkpoint_path, compute_dtype, config.weights_data_type, device
        )

        self.context_decoder.set_weight(self.weight)
        self.decoder.set_weight(self.weight)

        weight_dtype = to_numpy_dtype(config.weights_data_type)

        def _safe_load_from_bin(param: torch.nn.Parameter, fname):
            if (checkpoint_path / fname).exists():
                # np_w is 1-D array since a bin file doesn't have shape info.
                w_ = np.fromfile(checkpoint_path / fname, dtype=weight_dtype)
                param.data = (
                    torch.from_numpy(w_)
                    .reshape(param.data.shape)
                    .to(compute_dtype)
                )
            else:
                raise FileNotFoundError(f"Faile to load {fname}")

        def _safe_load_lm_head_from_bin(param, fname, ctype):
            if (checkpoint_path / fname).exists():
                shape = (
                    self.vocab_size,
                    self.config.head_num * self.config.size_per_head,
                )
                # np_w is 1-D array since a bin file doesn't have shape info.
                w_ = np.fromfile(checkpoint_path / fname, dtype=weight_dtype)
                param.data = param.data.to(ctype)
                param.data[: self.vocab_size, :] = (
                    torch.from_numpy(w_).reshape(shape).to(ctype)
                )
            else:
                print(f"Faile to load {fname}")
                torch.nn.init.normal_(param).to(compute_dtype)
            self._mask_padded_vocab_weights(param)

        # pylint:disable=line-too-long
        if comm.is_pipeline_group_first():
            _safe_load_lm_head_from_bin(
                self.word_embedding.weight, "model.wte.bin", compute_dtype
            )
            self._mask_padded_vocab_weights(self.word_embedding.weight)
            if self.position_encoding is not None:
                _safe_load_from_bin(
                    self.position_encoding.weight, "model.wpe.bin"
                )
            if self.pre_decoder_layernorm is not None:
                _safe_load_from_bin(
                    self.pre_decoder_layernorm.weight,
                    "model.pre_decoder_layernorm.weight.bin",
                )
                _safe_load_from_bin(
                    self.pre_decoder_layernorm.bias,
                    "model.pre_decoder_layernorm.bias.bin",
                )
        if comm.is_pipeline_group_last():
            if self.post_decoder_layernorm is not None:
                _safe_load_from_bin(
                    self.post_decoder_layernorm.weight,
                    "model.final_layernorm.weight.bin",
                )
                _safe_load_from_bin(
                    self.post_decoder_layernorm.bias,
                    "model.final_layernorm.bias.bin",
                )
            if (checkpoint_path / "model.lm_head.weight.bin").exists():
                _safe_load_lm_head_from_bin(
                    self.lm_head.weight,
                    "model.lm_head.weight.bin",
                    self.lm_head_ctype,
                )
            else:
                if self.use_fp32_to_compute_logit:
                    _safe_load_lm_head_from_bin(
                        self.lm_head.weight, "model.wte.bin", torch.float32
                    )
                else:
                    # In this branch we can share the pre and post
                    # decoder embeddings, but ONLY pipeline size is 1.
                    # When pipeline size > 1, these two weights will end up on
                    # different GPUs, so we must load the
                    # post decoder weight again (else case).
                    if comm.get_pipeline_para_size() == 1:
                        self.lm_head.weight = self.word_embedding.weight
                    else:
                        _safe_load_lm_head_from_bin(
                            self.lm_head.weight, "model.wte.bin", compute_dtype
                        )

        self.to(device)

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device

    def cuda(self, device=None):
        assert torch.cuda.is_available()
        for name, param in self._parameters.items():
            setattr(self, name, param.cuda(device))
        return self

    def to(self, device=None):
        for name, param in self._parameters.items():
            setattr(self, name, param.to(device))
        return self

    def float(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.float())
        return self

    def half(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.half())
        return self

    def bfloat16(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.bfloat16())
        return self

    def _mask_padded_vocab_weights(self, weight: torch.Tensor):
        assert self.vocab_size_padded >= self.vocab_size
        if self.vocab_size_padded > self.vocab_size:
            weight.data[self.vocab_size :, ...] = 0  # noqa: E203

    def generate_pad_mask(self, input_lengths, memory_length, init_step=0):
        """Generate a pad mask tensor.

        # Args.
            input_lengths: (batch_size * beam_width,), input lengths
            memory_length: the length of key/value cache memory.
            init_step: int, initial step.
        # Return
            masked_tokens: BoolTensor,
                (batch_size * beam_width, memory_length),
                True if init_step + input_length[i] <= j <
                    init_step + max_input_length,
                where i is a batch-beam index and j is a time step
                modulo by memory_length.
        """
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.unsqueeze(1)
        shift = init_step % memory_length
        step_indices = torch.arange(
            init_step, init_step + memory_length, device=input_lengths.device
        )
        step_indices = (
            step_indices.roll(shift)
            .unsqueeze(0)
            .tile(input_lengths.shape[0], 1)
        )
        masked_tokens = torch.logical_and(
            step_indices >= input_lengths,
            step_indices < init_step + max_input_length,
        )
        return masked_tokens

    def get_local_batch_size(self, batch_size):
        """Get a local batch size by the same way that FT Gpt does."""
        local_batch_size = batch_size
        pp_size = self.decoder.pipeline_para_size
        if pp_size > 1:
            if local_batch_size % pp_size == 0:
                local_batch_size //= pp_size
            while local_batch_size > 1024 and local_batch_size % 2 == 0:
                local_batch_size //= 2
        return local_batch_size

    @torch.no_grad()
    def generate(
        self,
        input_token_ids: torch.IntTensor,
        input_lengths: torch.IntTensor,
        gen_length: int,
        eos_token_id: Optional[int] = None,
        local_batch_size: Optional[int] = None,
        beam_width: int = 1,
        top_k: Optional[torch.IntTensor] = None,
        top_p: Optional[torch.FloatTensor] = None,
        top_p_decay: Optional[torch.FloatTensor] = None,
        top_p_min: Optional[torch.FloatTensor] = None,
        top_p_reset_ids: Optional[torch.IntTensor] = None,
        temperature: Optional[torch.FloatTensor] = None,
        repetition_penalty: Optional[torch.FloatTensor] = None,
        presence_penalty: Optional[torch.FloatTensor] = None,
        min_length: Optional[torch.IntTensor] = None,
        len_penalty: Optional[torch.FloatTensor] = None,
        beam_search_diversity_rate: Optional[torch.FloatTensor] = None,
        stop_words_list: Optional[torch.IntTensor] = None,
        bad_words_list: Optional[torch.IntTensor] = None,
        sequence_limit_lengths: Optional[torch.IntTensor] = None,
        random_seed: Optional[torch.LongTensor] = None,
        memory_length: Optional[int] = None,
        return_output_length: bool = False,
        return_log_probs: bool = False,
    ):
        """

        # Args.
            input_token_ids: IntTensor, (batch_size, max_input_length),
                input hidden state to decoder.
            input_lengths: IntTensor, (batch_size),
                the lengths of input context sequences.
            gen_length: int, the number of tokens to generate.
            local_batch_size: int, optional, a batch size of
                local iteration. (disabled)
            eos_token_id: int, eos token id.
            beam_width: int, number of beams for beam search.
                If 1, sampling decode will be used.
            top_k: IntTensor, (batch_size,) top-k sampling.
                The number of most probable tokens to keep
                for sampling per sentence in a batcch.
            top_p: FloatTensor, (batch_size,), top-p sampling.
                The cumulative probability
                of to filter the set of most probable tokens.
            top_p_decay: FloatTensor, (batch_size,)
                The decay of top-p value for top_p sampling.
            top_p_min: FloatTensor, (batch_size,)
                The minimum top p values in top-p decaying.
            top_p_reset_ids: IntTensor, (batch_size,)
                reset ids for resetting top_p values for top p sampling
            temperature: FloatTensor, (batch_size,),
                The temperature value for smoothing the logit distribution.
            repetition_penalty: FloatTensor, (batch_size,),
                The repetition penalty.
            presence_penalty: FloatTensor, (batch_size,),
                The presence penalty, which is exclusive with
                repetition_penalty.
                Only one of repetition and presence penalties is allowed.
            min_length: IntTensor, (batch_size,),
                Minimum length for each sentences. EOS is masked if length is
                below min.
            len_penalty: FloatTensor, (batch_size,)
                The exponent of the length penalty of beam scores.
            beam_search_diversity_rate: FloatTensor, (batch_size,),
                The diversity rate of beam search.
            stop_words_list: IntTensor, (batch_size, 2, stop_words_length)
                When FT generates words in this list, it will stop the
                generation. An extension of stop id.
            bad_words_list IntTensor, (batch_size, 2, bad_words_length)
                The words in the list will never be sampled.
            sequence_limit_lengths: IntTensor, (batch_size,), The maximum
                length of a generated sequence.
            memory_length: int, the length of cache memory. If None, it will
                be max_input_length + gen_length.
        # Returns
            IntTensor, (batch_size, beam_width, max_seq_length) output
            token ids.
        """
        assert (
            self.weight is not None
        ), "Please call load() first to initialize weights."

        input_token_ids = input_token_ids.type(torch.int32).to(self.device)
        input_lengths = input_lengths.type(torch.int32).to(self.device)

        batch_size = len(input_token_ids)
        max_input_length = input_token_ids.shape[-1]
        max_seq_length = max_input_length + gen_length
        memory_length = memory_length or max_seq_length

        # TODO: Enable local batch later. We currently disable local batching due to # noqa: E501
        #   an input mismatch issue of FT's decode_op: FT's decode_op requires logits # noqa: E501
        #   of shape (batch_size, ...) but we have logits of shape (local_batch_size, ...) # noqa: E501
        #   After fixing FT's side, we will enable local batch.
        # local_batch_size = local_batch_size or self.get_local_batch_size(batch_size) # noqa: E501
        # num_local_batches, last_chunk = divmod(batch_size, local_batch_size)
        # if last_chunk > 0:
        #     num_local_batches += 1
        assert local_batch_size is None or local_batch_size == batch_size
        local_batch_size = batch_size
        num_local_batches = 1

        device = self.device

        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.end_id
        )
        assert (
            eos_token_id is not None
        ), "eos_token-id must be specified in generation."
        eos_token_ids = eos_token_id * torch.ones(
            batch_size, dtype=torch.int32, device=device
        )
        assert repetition_penalty is None or presence_penalty is None, (
            "Found ambiguous parameters repetition_penalty and "
            "presence_penalty which are mutually exclusive. "
            "Please provide one of repetition_penalty and presence_penalty."
        )
        # Setup decoder_op prior to calling the forward function.
        self.decode_op.setup(
            batch_size,
            beam_width,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            presence_penalty,
            min_length,
            len_penalty,
            beam_search_diversity_rate,
            random_seed,
            top_p_decay,
            top_p_min,
            top_p_reset_ids,
        )

        # Prepare input and output arguments.
        if beam_width > 1:
            # Tiling for beam search.
            input_token_ids = input_token_ids.repeat(1, beam_width).view(
                batch_size * beam_width, -1
            )
            input_lengths = (
                input_lengths.view(-1, 1).repeat(1, beam_width).view(-1)
            )
            if sequence_limit_lengths is not None:
                sequence_limit_lengths = (
                    sequence_limit_lengths.view(-1, 1)
                    .repeat(1, beam_width)
                    .view(-1)
                )
            # src/tgt cache indirections.
            cache_indirection = torch.zeros(
                (2, batch_size, beam_width, memory_length),
                dtype=torch.int32,
                device=device,
            )
            parent_ids = torch.zeros(
                max_seq_length,
                batch_size * beam_width,
                dtype=torch.int32,
                device=device,
            )
        else:
            cache_indirection = None
            src_cache_indirection = None
            tgt_cache_indirection = None
            parent_ids = None

        pad_lengths = max_input_length - input_lengths
        # Since tril() doesn't support bf16 dtype,
        # we create of bool type and then cast it to dtype.
        attention_mask = (
            torch.ones(
                (max_input_length, max_input_length),
                dtype=torch.bool,
                device=device,
            )
            .tril()
            .unsqueeze(0)
            .tile(input_token_ids.shape[0], 1, 1)
            .to(self.dtype)
        )
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
        masked_tokens = self.generate_pad_mask(input_lengths, memory_length)
        finished = torch.zeros_like(input_lengths).bool()
        sequence_lengths = (max_input_length - 1) * torch.ones_like(
            input_lengths
        )

        if return_log_probs or beam_width > 1:
            cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
            output_log_probs = torch.zeros(
                (gen_length, batch_size * beam_width), device=device
            )
        else:
            cum_log_probs = None
            output_log_probs = None

        # Contiguous buffer for each decode_op step,
        # it will be transposed tensor for the final output.
        output_token_ids = torch.zeros(
            (max_seq_length, batch_size * beam_width),
            dtype=torch.int32,
            device=device,
        )
        output_token_ids[:max_input_length, ...] = input_token_ids.T

        if comm.is_pipeline_group_first():
            # Prepare input tensors of decoder.
            input_embeds = self.word_embedding(input_token_ids)
            if self.position_encoding is not None:
                position_ids = torch.arange(
                    0, max_input_length, dtype=torch.int, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(
                    -1, max_input_length
                )
                input_embeds += self.position_encoding(position_ids)
            if self.pre_decoder_layernorm is not None:
                input_embeds = self.pre_decoder_layernorm(input_embeds)
        else:
            # Dummy input_embeds
            input_embeds = torch.empty(
                size=(
                    batch_size * beam_width,
                    max_input_length,
                    self.context_decoder.hidden_size,
                ),
                dtype=self.context_decoder.dtype,
                device=device,
            )

        use_shared_contexts = (
            (self.shared_contexts_ratio > 0.0)
            and (max_input_length >= 1)
            and (batch_size > 1)
        )
        batch_to_compact, compact_to_batch = None, None
        if use_shared_contexts:
            find_context_duplications = (
                torch.ops.fastertransformer.find_context_duplications
            )
            batch_to_compact, compact_to_batch = find_context_duplications(
                input_token_ids
            )
            use_shared_contexts = (
                compact_to_batch.shape[0]
                <= self.shared_contexts_ratio * batch_size
            )

            if not use_shared_contexts:
                batch_to_compact, compact_to_batch = None, None

        profiler.start("ft-context-decoder")
        (
            _,
            k_cache,
            v_cache,
            last_token_hidden_states,
        ) = self.context_decoder.forward(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            memory_length=memory_length,
            batch_to_compact_index=batch_to_compact,
            compact_index=compact_to_batch,
        )
        profiler.stop("ft-context-decoder")

        for step in range(max_input_length, max_seq_length):
            src_indir_idx = (step - max_input_length) % 2
            tgt_indir_idx = 1 - src_indir_idx

            is_generation_done = torch.tensor(
                [True], dtype=torch.bool, device=device
            )
            for ite in range(num_local_batches):
                # The indices of the current local batch-beam.
                bbidx = range(
                    ite * local_batch_size * beam_width,
                    min(
                        (ite + 1) * local_batch_size * beam_width,
                        batch_size * beam_width,
                    ),
                )
                if cache_indirection is not None:
                    bidx = range(
                        ite * local_batch_size,
                        min((ite + 1) * local_batch_size, batch_size),
                    )
                    src_cache_indirection = cache_indirection[
                        src_indir_idx, bidx, ...
                    ]
                    tgt_cache_indirection = cache_indirection[
                        tgt_indir_idx, bidx, ...
                    ]

                if step == max_input_length:
                    hidden_states = last_token_hidden_states[bbidx, ...]
                else:
                    if comm.is_pipeline_group_first():
                        input_embeds = self.word_embedding(
                            output_token_ids[step - 1, bbidx]
                        )
                        if self.position_encoding is not None:
                            position_ids = (step - 1) * torch.ones_like(
                                pad_lengths[bbidx]
                            )
                            input_embeds += self.position_encoding(
                                position_ids
                            )
                        if self.pre_decoder_layernorm is not None:
                            input_embeds = self.pre_decoder_layernorm(
                                input_embeds
                            )
                    else:
                        # Dummy input_imbeds
                        input_embeds = torch.empty(
                            size=(len(bbidx), self.decoder.hidden_size),
                            dtype=self.decoder.dtype,
                            device=device,
                        )

                    profiler.start("ft-decoder")
                    hidden_states = self.decoder.forward(
                        max_input_length=max_input_length,
                        step=step,
                        ite=ite,
                        input_embeds=input_embeds,
                        sequence_lengths=sequence_lengths[bbidx],
                        key_cache=k_cache,
                        value_cache=v_cache,
                        finished=finished[bbidx],
                        total_padding_tokens=pad_lengths[bbidx],
                        cache_indirection=src_cache_indirection,
                        masked_tokens=masked_tokens[bbidx, ...],
                    )
                    profiler.stop("ft-decoder")

                if comm.is_pipeline_group_last():
                    if self.post_decoder_layernorm is not None:
                        hidden_states = self.post_decoder_layernorm(
                            hidden_states
                        )

                    # We use logits of fp32 type to avoid overflow issue.
                    if self.use_fp32_to_compute_logit:
                        # The FT GPT op internally uses FP32 compute type
                        # for matrix multiplication.
                        # This will produce the same result with the
                        # end-to-end FT's GPT op.
                        logits = torch.nn.functional.linear(
                            hidden_states.float(), self.lm_head.weight
                        )
                    else:
                        logits = self.lm_head(hidden_states).float()

                    profiler.start("ft-decode")
                    should_stop = self.decode_op.forward(
                        logits.view(batch_size, beam_width, -1),
                        step,
                        max_input_length,
                        ite,
                        local_batch_size,
                        eos_token_ids,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                        presence_penalty,
                        min_length,
                        len_penalty,
                        beam_search_diversity_rate,
                        top_p_decay,
                        top_p_min,
                        top_p_reset_ids,
                        None,
                        input_lengths,
                        sequence_limit_lengths,
                        stop_words_list,
                        bad_words_list,
                        src_cache_indirection,
                        output_token_ids.view(-1, batch_size, beam_width),
                        finished,
                        sequence_lengths,
                        cum_log_probs,
                        output_log_probs,
                        parent_ids,
                        tgt_cache_indirection,
                    )
                    profiler.stop("ft-decode")
                    is_generation_done &= should_stop

            # Broadcast from the last pipeline node if needed.
            profiler.start("ft-bcast")
            tensors_to_bcast = [
                output_token_ids[step, ...],
                finished,
                sequence_lengths,
                is_generation_done,
            ]
            if beam_width > 1:
                tensors_to_bcast.append(tgt_cache_indirection)
            self.decode_op.broadcast_from_last_pipeline(tensors_to_bcast)
            profiler.stop("ft-bcast")

            if is_generation_done or finished.all():
                break

        # Transpose (L, batch, beam) -> (batch, beam, L)
        output_token_ids = output_token_ids.view(
            -1, batch_size, beam_width
        ).permute(1, 2, 0)

        # Increase sequence_length by 1 because the sequence length of time step t is t - 1. # noqa: E501
        sequence_lengths += 1

        # Outputs
        output_dict = dict(output_token_ids=output_token_ids)
        if return_output_length:
            output_dict["output_lengths"] = sequence_lengths
        if return_log_probs:
            output_dict["cum_log_probs"] = cum_log_probs
            output_dict["output_log_probs"] = output_log_probs
        return output_dict
