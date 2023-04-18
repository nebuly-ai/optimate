# Based on: https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/bert/utils/modeling_bert.py  # noqa: E501
# This file is mostly copied from the FasterTransformer repo
# https://github.com/NVIDIA/FasterTransformer
# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from loguru import logger

from nebullvm.optional_modules.torch import torch, torch_distributed as dist

from nebullvm.optional_modules.huggingface import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)

from .checkpoint_quantization import checkpoint_quantization


class EncoderWeights(object):
    def __init__(
        self,
        layer_num,
        hidden_dim,
        weights=None,
        sparse=False,
        tensor_para_size=1,
        pipeline_para_size=1,
    ):
        """weights need be a state_dict of bert model"""
        self.layer_num = layer_num
        self.int8 = False
        self.hidden_dim = hidden_dim
        self.weights = {}
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size

        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend="mpi")
            except:  # noqa: E722
                logger.info(
                    "[INFO] WARNING: Exception occurred in "
                    "dist.init_process_group(backend='mpi')."
                    "Maybe the process group has been initialized somewhere else."  # noqa: E501
                )
        else:
            logger.info("[INFO] MPI is not available in this PyTorch build.")
            assert (
                tensor_para_size == 1
            ), "[FATAL] MPI is required for tensor_para_size > 1."
            assert (
                pipeline_para_size == 1
            ), "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1  # noqa: F841
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size
        if weights is None:
            self._generated_weights = True
            for i in range(layer_num):
                pre = "encoder.layer." + str(i) + "."
                self.weights[
                    pre + "attention.self.query.weight"
                ] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + "attention.self.query.bias"] = torch.zeros(
                    hidden_dim
                )
                self.weights[pre + "attention.self.key.weight"] = torch.zeros(
                    hidden_dim, hidden_dim
                )
                self.weights[pre + "attention.self.key.bias"] = torch.zeros(
                    hidden_dim
                )
                self.weights[
                    pre + "attention.self.value.weight"
                ] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + "attention.self.value.bias"] = torch.zeros(
                    hidden_dim
                )
                self.weights[
                    pre + "attention.output.dense.weight"
                ] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[
                    pre + "attention.output.dense.bias"
                ] = torch.zeros(hidden_dim)
                self.weights[
                    pre + "attention.output.LayerNorm.weight"
                ] = torch.zeros(hidden_dim)
                self.weights[
                    pre + "attention.output.LayerNorm.bias"
                ] = torch.zeros(hidden_dim)
                self.weights[pre + "intermediate.dense.weight"] = torch.zeros(
                    4 * hidden_dim, hidden_dim
                )  # noqa: E501
                self.weights[pre + "intermediate.dense.bias"] = torch.zeros(
                    4 * hidden_dim
                )
                self.weights[pre + "output.dense.weight"] = torch.zeros(
                    hidden_dim, 4 * hidden_dim
                )
                self.weights[pre + "output.dense.bias"] = torch.zeros(
                    hidden_dim
                )
                self.weights[pre + "output.LayerNorm.weight"] = torch.zeros(
                    hidden_dim
                )
                self.weights[pre + "output.LayerNorm.bias"] = torch.zeros(
                    hidden_dim
                )
            for k, v in self.weights.items():
                if not k.endswith("_amax"):
                    self.weights[k] = torch.nn.init.uniform_(v, -1, 1)
            if sparse:
                for k, v in self.weights.items():
                    if (
                        "query.weight" in k
                        or "key.weight" in k
                        or "value.weight" in k
                        or "dense.weight" in k
                    ):
                        v_shape = v.shape
                        v = v.view(-1, 4)
                        _, indices = torch.topk(
                            torch.abs(v), 2, dim=-1, largest=False
                        )
                        v.scatter_(1, indices, 0)
                        self.weights[k] = v.view(v_shape)
        else:
            self._generated_weights = False
            for k, v in weights.items():
                ks = k.split(".")
                if ks[-2] == "LayerNorm":
                    if ks[-1] == "gamma":
                        ks[-1] = "weight"
                    elif ks[-1] == "beta":
                        ks[-1] = "bias"
                self.weights[".".join(ks)] = v

    def listed_weights(self):
        ret = []
        start_layer = (
            self.pipeline_para_rank * self.layer_num // self.pipeline_para_size
        )
        end_layer = (
            (self.pipeline_para_rank + 1)
            * self.layer_num
            // self.pipeline_para_size
        )
        if not self.int8:
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.query.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 0
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.query.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.key.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 2
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.key.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.value.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 4
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.value.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.dense.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 6
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[1] // self.tensor_para_size, dim=1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.dense.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.LayerNorm.weight"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.LayerNorm.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "intermediate.dense.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 10
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "intermediate.dense.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[-1] // self.tensor_para_size, dim=-1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.dense.weight"
                        ].transpose(-1, -2)
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )  # 12
            ret[-1] = (
                ret[-1]
                .split(ret[-1].shape[1] // self.tensor_para_size, dim=1)[
                    self.tensor_para_rank
                ]
                .contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.dense.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.LayerNorm.weight"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.LayerNorm.bias"
                        ]
                        for layer_idx in range(start_layer, end_layer)
                    ],
                    0,
                ).contiguous()
            )
        else:
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.query.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 0
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.query.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.key.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 2
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.key.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.value.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 4
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.self.value.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.dense.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 6
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.dense.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.LayerNorm.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "attention.output.LayerNorm.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "intermediate.dense.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 10
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "intermediate.dense.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.dense.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )  # 12
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.dense.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.LayerNorm.weight"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "output.LayerNorm.bias"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "amaxList"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
            ret.append(
                torch.stack(
                    [
                        self.weights[
                            "encoder.layer."
                            + str(layer_idx)
                            + "."
                            + "h_amaxList"
                        ]
                        for layer_idx in range(self.layer_num)
                    ],
                    0,
                ).contiguous()
            )
        return ret

    def to_cuda(self):
        if not self.int8:
            for k, v in self.weights.items():
                self.weights[k] = v.cuda()
        else:
            h_scale_list = {}
            for k, v in self.weights.items():
                if "amaxList" in k:
                    k_h = k.replace("amaxList", "h_amaxList")
                    h_scale_list[k_h] = v
                self.weights[k] = v.cuda()
            for k, v in h_scale_list.items():
                self.weights[k] = v

    def to_half(self):
        if self.int8:
            raise RuntimeError(
                "Cannot cast to half if the weights have been casted to int8."
            )
        for k, v in self.weights.items():
            self.weights[k] = v.half()

    def to_bfloat16(self):
        if self.int8:
            raise RuntimeError(
                "Cannot cast to bfloat16 if the weights have been casted to int8."  # noqa: E501
            )
        for k, v in self.weights.items():
            self.weights[k] = v.bfloat16()

    def to_int8(self, sparse=False, ths_path="./lib/libth_transformer.so"):
        if self._generated_weights:
            amax_tensor_1 = torch.Tensor(self.hidden_dim).fill_(127.0)
            amax_tensor_2 = torch.Tensor(self.hidden_dim * 4).fill_(127.0)
            for i in range(self.layer_num):
                pre = "encoder.layer." + str(i) + "."
                self.weights[
                    pre + "attention.self.query._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.query._weight_quantizer._amax"
                ] = amax_tensor_1
                self.weights[
                    pre + "attention.self.query._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.key._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.key._weight_quantizer._amax"
                ] = amax_tensor_1
                self.weights[
                    pre + "attention.self.key._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.value._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.value._weight_quantizer._amax"
                ] = amax_tensor_1
                self.weights[
                    pre + "attention.self.value._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.matmul_q_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.matmul_k_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.matmul_v_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.matmul_a_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.self.softmax_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.output.dense._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.output.dense._weight_quantizer._amax"
                ] = amax_tensor_1
                self.weights[
                    pre + "attention.output.dense._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.output.add_local_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "attention.output.add_residual_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "intermediate.dense._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "intermediate.dense._weight_quantizer._amax"
                ] = amax_tensor_2
                self.weights[
                    pre + "intermediate.dense._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "output.dense._input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "output.dense._weight_quantizer._amax"
                ] = amax_tensor_1
                self.weights[
                    pre + "output.dense._aftergemm_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "output.add_local_input_quantizer._amax"
                ] = torch.tensor(127.0)
                self.weights[
                    pre + "output.add_residual_input_quantizer._amax"
                ] = torch.tensor(127.0)
        if (
            "encoder.layer.0.attention.self.query._input_quantizer._amax"
            not in self.weights
        ):
            raise RuntimeError(
                "There is no quantization node in the checkpoint, cannot be quantized to int8."  # noqa: E501
            )
        if self.int8:
            return
        self.int8 = True
        for k, v in self.weights.items():
            if k.endswith("bias") or k.endswith("LayerNorm.weight"):
                self.weights[k] = v.half()
            elif k.endswith("weight"):
                self.weights[k] = v.float().cuda()
            else:
                self.weights[k] = v.float().cpu()
        self.weights = checkpoint_quantization(
            self.weights, sparse, ths_path, verbose=False
        )


class CustomEncoder(torch.nn.Module):
    def __init__(
        self,
        layer_num,
        head_num,
        head_size,
        weights,
        int8_mode=0,
        remove_padding=False,
        sparse=False,
        path="./lib/libth_transformer.so",
        tensor_para_size=1,
        pipeline_para_size=1,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.remove_padding = remove_padding
        self.int8_mode = int8_mode
        logger.info(f"loading faster transformer library from {path}")
        torch.classes.load_library(path)

        weights_ = weights.listed_weights()

        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend="mpi")
            except:  # noqa: E722
                logger.info(
                    "[INFO] WARNING: Exception occurred in"
                    "dist.init_process_group(backend='mpi')."
                    "Maybe the process group has been initialized somewhere else."  # noqa: E501
                )
        else:
            logger.info("[INFO] MPI is not available in this PyTorch build.")
            assert (
                tensor_para_size == 1
            ), "[FATAL] MPI is required for tensor_para_size > 1."
            assert (
                pipeline_para_size == 1
            ), "[FATAL] MPI is required for pipeline_para_size > 1."

        if int8_mode == 0:
            assert len(weights_) == 16
            try:
                self.encoders = torch.classes.FasterTransformer.Bert(
                    *weights_,
                    head_num,
                    head_size,
                    4 * head_num * head_size,
                    remove_padding,
                    layer_num,
                    sparse,
                    1.0,
                    tensor_para_size,
                    pipeline_para_size,
                )
            except:  # noqa: E722
                # legacy ths for 20.03 image
                self.encoders = torch.classes.FasterTransformerBert(
                    *weights_,
                    head_num,
                    head_size,
                    4 * head_num * head_size,
                    remove_padding,
                    layer_num,
                    sparse,
                    1.0,
                    tensor_para_size,
                    pipeline_para_size,
                )
        else:
            assert len(weights_) == 18
            assert (
                tensor_para_size == 1
            ), "INT8 BERT still only support tensor_para_size = 1"
            assert (
                pipeline_para_size == 1
            ), "INT8 BERT still only support pipeline_para_size = 1"
            try:
                self.encoders = torch.classes.FasterTransformer.INT8Bert(
                    *weights_,
                    head_num,
                    head_size,
                    remove_padding,
                    layer_num,
                    int8_mode,
                    sparse,
                    1.0,
                )
            except:  # noqa: E722
                # legacy ths for 20.03 image
                self.encoders = torch.classes.FasterTransformerINT8Bert(
                    *weights_,
                    head_num,
                    head_size,
                    remove_padding,
                    layer_num,
                    int8_mode,
                    sparse,
                    1.0,
                )

    def forward(self, hidden_states, attention_mask, sequence_lengths):
        hidden_states = self.encoders.forward(hidden_states, sequence_lengths)
        return (hidden_states,)


class HuggingFaceEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights=None):
        super().__init__()
        hidden_dim = head_num * head_size
        # TODO(bhsueh) The implementation of hidden_act='gelu' is differen
        #  to FT's (and google BERT) implementation
        # FT's implementation is equivalent to hidden_act='gelu_new',
        # but there are some issues for int8 sparse under gelu_new
        conf = BertConfig(
            hidden_size=hidden_dim,
            intermediate_size=4 * hidden_dim,
            num_attention_heads=head_num,
            num_hidden_layers=layer_num,
            hidden_act="gelu",
        )
        self.encoder = BertEncoder(conf)
        w = {}
        for k, v in weights.weights.items():
            if k.startswith("encoder") and not k.endswith("_amax"):
                w[k[13:]] = weights.weights[k]
        self.encoder.load_state_dict(w)
        self.head_mask = [None] * layer_num

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        output = self.encoder(
            hidden_states,
            extended_attention_mask,
            self.head_mask,
            return_dict=False,
        )
        return output


# Based on: https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/bert/utils/modeling_bert.py # noqa: E501
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team. # noqa: E501
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
"""PyTorch BERT model modified from HuggingFace transformers. """


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()
        self.use_ext_encoder = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"  # noqa: E501
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device
            )

        if self.use_ext_encoder:
            # if attention_mask.dim() == 3:
            #     extended_attention_mask = attention_mask
            # elif attention_mask.dim() == 2:
            #     extended_attention_mask = attention_mask[:, None, :].repeat(1, input_shape[1], 1) # noqa: E501
            # else:
            #     raise ValueError(
            #         "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(# noqa: E501
            #             input_shape, attention_mask.shape
            #         )
            #     )
            assert attention_mask.dim() == 2
            extended_attention_mask = attention_mask.view(
                -1, 1, 1, attention_mask.size(-1)
            )
            m_2 = extended_attention_mask.transpose(-1, -2)
            extended_attention_mask = extended_attention_mask * m_2
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            seq_lens = torch.sum(attention_mask, 1, dtype=torch.int32).cuda()
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length] # noqa: E501
            # ourselves in which case we just need to make it broadcastable to all heads. # noqa: E501
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(  # noqa: E501
                        input_shape, attention_mask.shape
                    )
                )
            # Since attention_mask is 1.0 for positions we want to attend
            # and 0.0 for masked positions, this operation will create a
            # tensor which is 0.0 for positions we want to attend
            # and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax,
            # this is effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_attention_mask = (
                1.0 - extended_attention_mask
            ) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        if self.use_ext_encoder:
            encoder_outputs = self.encoder(
                embedding_output, extended_attention_mask, seq_lens
            )
        else:
            head_mask = [None] * self.config.num_hidden_layers
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions) # noqa: E501

    def replace_encoder(self, new_encoder):
        self.encoder = new_encoder
        self.use_ext_encoder = True
