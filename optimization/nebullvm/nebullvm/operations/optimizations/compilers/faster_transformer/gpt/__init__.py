# Based on: https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/gpt/gpt_summarization.py # noqa: E501
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

import os
import tempfile
from typing import Callable, Iterable, List, Optional, Tuple, Union

from nebullvm.operations.optimizations.compilers.faster_transformer.gpt.utils import \
    gpt_decoder
from nebullvm.operations.optimizations.compilers.faster_transformer.gpt.utils.huggingface_gpt_convert import (  # noqa: E501
    main as convert_huggingface_gpt_to_faster_transformer,
)
from nebullvm.operations.optimizations.compilers.utils import (
    get_faster_transformer_repo_path,
)
from nebullvm.optional_modules.huggingface import GPT2LMHeadModel
from nebullvm.optional_modules.torch import torch

lib_path = default_lib_path = str(
    get_faster_transformer_repo_path()
    / "build"
    / "lib"
    / "libth_transformer.so"
)


class FasterTransformerGPT2Wrapper(torch.nn.Module):
    def __init__(self, model: gpt_decoder.Gpt, config):
        super().__init__()
        self.model = model
        self.config = config
        self.device = model.device

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = 1,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[
            Union[Iterable[int], Iterable[Iterable[int]]]
        ] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        # logits_processor: Optional[LogitsProcessorList] = None,
        # renormalize_logits: Optional[bool] = None,
        # stopping_criteria: Optional[StoppingCriteriaList] = None,
        # constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
    ):

        input_lengths = torch.tensor(
            [len(input) for input in inputs],
            dtype=torch.int32,
            device=self.model.device,
        )
        batch_size = len(inputs)

        def convert_to_tensor_if_not(value, dtype=torch.float32):
            if value is None:
                return value
            if isinstance(value, torch.Tensor):
                return value
            return value * torch.ones(batch_size, dtype=dtype)  # cpu tensor

        top_k = convert_to_tensor_if_not(top_k, dtype=torch.int32)
        top_p = convert_to_tensor_if_not(top_p, dtype=torch.float32)
        temperature = convert_to_tensor_if_not(
            temperature, dtype=torch.float32
        )
        repetition_penalty = convert_to_tensor_if_not(
            repetition_penalty, dtype=torch.float32
        )
        min_length = convert_to_tensor_if_not(min_length, dtype=torch.int32)
        len_penalty = convert_to_tensor_if_not(
            length_penalty, dtype=torch.float32
        )
        if max_length is None:
            # gen_length is required for faster transformer
            # infer it from the model config
            max_length = self.config.n_ctx
        output_dict = self.model.generate(
            input_token_ids=inputs,
            input_lengths=input_lengths,
            gen_length=max_length - len(inputs[0]),
            eos_token_id=eos_token_id,
            # local_batch_size=None,
            beam_width=num_beams,
            top_k=top_k,
            top_p=top_p,
            # top_p_decay: Optional[torch.FloatTensor] = None,
            # top_p_min: Optional[torch.FloatTensor] = None,
            # top_p_reset_ids: Optional[torch.IntTensor] = None,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            # presence_penalty: Optional[torch.FloatTensor] = None,
            min_length=min_length,
            len_penalty=len_penalty,
            # beam_search_diversity_rate: Optional[torch.FloatTensor] = None,
            # stop_words_list: Optional[torch.IntTensor] = None,
            # bad_words_list: Optional[torch.IntTensor] = None,
            # sequence_limit_lengths: Optional[torch.IntTensor] = None,
            # random_seed: Optional[torch.LongTensor] = None,
            # memory_length: Optional[int] = None,
            return_output_length=True,
            return_log_probs=False,
        )
        output_token_ids = output_dict["output_token_ids"]
        output_lengths = output_dict["output_lengths"]
        # tokens = output_token_ids[0, 0, input_lengths[0]:output_lengths[0]]
        tokens = [
            # output_token_ids[i, 0, input_lengths[i]:output_lengths[i]]
            output_token_ids[i, 0, : output_lengths[i]]
            for i in range(batch_size)
        ]
        return tokens


def convert_gpt2_lm_head_model(
    model: GPT2LMHeadModel,
    tokenizer,
    weight_data_type="fp32",
    data_type="fp16",
    use_fp32_to_compute_logit=False,
):
    """
    currently doens't support fp8 or multi-gpu
    """
    weights_data_type = weight_data_type
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    ft_model_location = saved_dir = temp_dir_path + "/gpt2"
    hf_config = model.config.to_dict()
    # convert huggingface model to faster transformer model
    convert_huggingface_gpt_to_faster_transformer(
        saved_dir=saved_dir,
        model=model.transformer,
        weight_data_type=weight_data_type,
    )

    head_num = hf_config["n_head"]
    layer_num = hf_config["n_layer"]
    start_id = hf_config["bos_token_id"]
    end_id = hf_config["eos_token_id"]
    size_per_head = hf_config["n_embd"] // head_num

    vocab_size = tokenizer.vocab_size

    tensor_para_size = 1
    pipeline_para_size = 1
    ckpt_path = os.path.join(ft_model_location, f"{tensor_para_size}-gpu")
    max_seq_len = hf_config["n_ctx"]
    int8_mode = 0  # 0: no quantization, 1: quantize weights to int8
    # load faster transformer model, note that the lm_head is not saved
    # it's reconstructed during loading from the embedding weights
    gpt = gpt_decoder.Gpt(
        num_heads=head_num,
        size_per_head=size_per_head,
        num_layers=layer_num,
        vocab_size=vocab_size,
        start_id=start_id,
        end_id=end_id,
        tensor_para_size=tensor_para_size,
        pipeline_para_size=pipeline_para_size,
        lib_path=lib_path,
        max_seq_len=max_seq_len,
        int8_mode=int8_mode,
        inference_data_type=data_type,
        weights_data_type=weights_data_type,
        use_fp32_to_compute_logit=use_fp32_to_compute_logit,
    )
    gpt.load(ckpt_path, data_type)
    return FasterTransformerGPT2Wrapper(gpt, model.config)


# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# model = hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda").eval()
# hf_config = hf_model.config.to_dict()


# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# weight_data_type = weights_data_type = "fp32" # fp32 or fp16
# data_type = "fp32" # fp32 or fp16
# faster_model= convert_gpt2_lm_head_model(
# model, tokenizer,
# weight_data_type=weight_data_type,
# data_type=data_type)
