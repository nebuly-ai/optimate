# Based on: https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/bert/utils/checkpoint_quantization.py # noqa: E501
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

import re

import numpy as np
from loguru import logger

from nebullvm.optional_modules.torch import torch

ACTIVATION_AMAX_NUM = 72
INT8O_GEMM_NUM = 8
TRT_FUSED_MHA_AMAX_NUM = 3
SCALE_RESERVE_NUM = 21


def checkpoint_quantization(
    init_dict, sparse, ths_path="./lib/libth_transformer.so"
):
    logger.info("Quantizing checkpoint ...")
    torch.classes.load_library(ths_path)
    weight_quantize = torch.ops.fastertransformer.weight_quantize

    def init_graph():
        layer_num = 0
        regex = re.compile("layer.\d+")  # noqa: W605
        amaxTotalNum = 0
        for name, tensor_value in init_dict.items():
            if "intermediate.dense.weight" in name and amaxTotalNum == 0:
                amaxTotalNum = (
                    ACTIVATION_AMAX_NUM
                    + 9 * tensor_value.size(1)
                    + INT8O_GEMM_NUM
                    + TRT_FUSED_MHA_AMAX_NUM
                    + SCALE_RESERVE_NUM
                )
            tmp = regex.findall(name)
            if len(tmp) < 1:
                continue
            num_tmp = int(tmp[0].replace("layer.", ""))
            if layer_num < num_tmp:
                layer_num = num_tmp
        layer_num = layer_num + 1
        # add new var for amax
        for i in range(layer_num):
            init_dict[
                "bert.encoder.layer.{}.amaxList".format(i)
            ] = torch.zeros((amaxTotalNum,), dtype=torch.float32)
        return layer_num, amaxTotalNum

    layer_num, amaxTotalNum = init_graph()

    kernel_name_list = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]

    amax_name_list = [
        "attention.self.query._input_quantizer",
        "attention.self.query._aftergemm_quantizer",
        "attention.self.matmul_q_input_quantizer",
        "attention.self.key._aftergemm_quantizer",
        "attention.self.matmul_k_input_quantizer",
        "attention.self.value._aftergemm_quantizer",
        "attention.self.matmul_v_input_quantizer",
        "attention.self.softmax_input_quantizer",
        "attention.self.matmul_a_input_quantizer",
        "attention.output.dense._input_quantizer",
        "attention.output.dense._aftergemm_quantizer",
        "intermediate.dense._input_quantizer",
        "intermediate.dense._aftergemm_quantizer",
        "output.dense._input_quantizer",
        "output.dense._aftergemm_quantizer",
        "special_F2Bias_scale",
    ]

    int8O_gemm_weight_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_weight_list = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.self.matmul_k_input_quantizer",
        "attention.self.matmul_v_input_quantizer",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]

    int8O_gemm_input_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_input_list = [
        "attention.self.query._input_quantizer",
        "attention.self.key._input_quantizer",
        "attention.self.value._input_quantizer",
        "attention.self.matmul_q_input_quantizer",
        "attention.self.matmul_a_input_quantizer",
        "attention.output.dense._input_quantizer",
        "intermediate.dense._input_quantizer",
        "output.dense._input_quantizer",
    ]

    int8O_gemm_output_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_output_list = [
        "attention.self.query._aftergemm_quantizer",
        "attention.self.key._aftergemm_quantizer",
        "attention.self.value._aftergemm_quantizer",
        "attention.self.softmax_input_quantizer",
        "attention.output.dense._input_quantizer",
        "attention.output.dense._aftergemm_quantizer",
        "intermediate.dense._aftergemm_quantizer",
        "output.dense._aftergemm_quantizer",
    ]

    same_value_tuple_list = [
        (
            "attention.self.query._input_quantizer",
            "attention.self.key._input_quantizer",
            "attention.self.value._input_quantizer",
            "attention.output.add_residual_input_quantizer",
        ),
        (
            "intermediate.dense._input_quantizer",
            "output.add_residual_input_quantizer",
        ),
    ]

    factor = 1000000.0  # noqa: F841
    for i in range(layer_num):
        amaxList = np.zeros([amaxTotalNum]).astype(np.float32)
        amax_id = 0
        # verify some quantizers have same value.
        # input_quantizer is per-tensor quantization
        for same_value_tuple in same_value_tuple_list:
            tmp_v = init_dict[
                "bert.encoder.layer.{}.{}._amax".format(i, same_value_tuple[0])
            ].numpy()
            for same_value_name in same_value_tuple:
                tmp_v_2 = init_dict[
                    "bert.encoder.layer.{}.{}._amax".format(i, same_value_name)
                ].numpy()
                assert np.allclose(tmp_v, tmp_v_2)

        for amax_name in amax_name_list:
            if amax_name == "special_F2Bias_scale":
                if i != layer_num - 1:
                    quant_max = init_dict[
                        "bert.encoder.layer.{}.{}._amax".format(
                            i + 1, amax_name_list[0]
                        )
                    ].item()
                    amax = abs(quant_max)
                else:
                    # not used, placeholder
                    amax = 1.0
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax / 127.0
                amax_id += 1
                amaxList[amax_id] = amax / 127.0 / 127.0
                amax_id += 1
                amaxList[amax_id] = 127.0 / amax
                amax_id += 1
                continue

            quant_max = init_dict[
                "bert.encoder.layer.{}.{}._amax".format(i, amax_name)
            ].item()
            amax = abs(quant_max)  # round(abs(quant_max)*factor)/factor
            if amax_name in int8O_gemm_input_list:
                int8O_gemm_input_amax_list[
                    int8O_gemm_input_list.index(amax_name)
                ] = amax
                if amax_name == "attention.self.query._input_quantizer":
                    int8O_gemm_input_amax_list[
                        int8O_gemm_input_list.index(
                            "attention.self.key._input_quantizer"
                        )
                    ] = amax
                    int8O_gemm_input_amax_list[
                        int8O_gemm_input_list.index(
                            "attention.self.value._input_quantizer"
                        )
                    ] = amax
            if amax_name in int8O_gemm_output_list:
                int8O_gemm_output_amax_list[
                    int8O_gemm_output_list.index(amax_name)
                ] = amax
            if amax_name in int8O_gemm_weight_list:
                int8O_gemm_weight_amax_list[
                    int8O_gemm_weight_list.index(amax_name)
                ] = amax
            amaxList[amax_id] = amax
            amax_id += 1
            amaxList[amax_id] = amax / 127.0
            amax_id += 1
            amaxList[amax_id] = amax / 127.0 / 127.0
            amax_id += 1
            amaxList[amax_id] = 127.0 / amax
            amax_id += 1

        # kernel amax starts from ACTIVATION_AMAX_NUM
        assert amax_id == 64
        amax_id = ACTIVATION_AMAX_NUM
        for kernel_id, kernel_name in enumerate(kernel_name_list):
            kernel = (
                init_dict[
                    "bert.encoder.layer.{}.{}.weight".format(i, kernel_name)
                ]
                .transpose(-1, -2)
                .contiguous()
            )
            quant_max2 = init_dict[
                "bert.encoder.layer.{}.{}._weight_quantizer._amax".format(
                    i, kernel_name
                )
            ]
            amax2 = abs(quant_max2)
            if amax2.dim() == 0:
                quant_max_processed = torch.full(
                    (kernel.size(1),),
                    amax2.item(),
                    dtype=amax2.dtype,
                    device=amax2.device,
                )
            else:
                quant_max_processed = amax2.view(-1)
            kernel_processed = weight_quantize(
                kernel, quant_max_processed.cuda(), sparse
            )
            init_dict[
                "bert.encoder.layer.{}.{}.weight".format(i, kernel_name)
            ] = kernel_processed
            if kernel_name in int8O_gemm_weight_list:
                int8O_gemm_weight_amax_list[
                    int8O_gemm_weight_list.index(kernel_name)
                ] = quant_max_processed[0]
            for e in quant_max_processed:
                amaxList[amax_id] = e
                amax_id += 1

        # for int8O gemm deQuant
        for j in range(INT8O_GEMM_NUM):
            amaxList[amax_id] = (
                int8O_gemm_input_amax_list[j] * int8O_gemm_weight_amax_list[j]
            ) / (127.0 * int8O_gemm_output_amax_list[j])
            amax_id += 1

        # for trt fused MHA amax
        # QKV_addBias_amax
        amaxList[amax_id] = np.maximum(
            np.maximum(amaxList[8], amaxList[16]), amaxList[24]
        )
        amax_id += 1
        # softmax amax
        amaxList[amax_id] = amaxList[32]
        amax_id += 1
        # bmm2 amax
        amaxList[amax_id] = amaxList[36]
        amax_id += 1

        init_dict["bert.encoder.layer.{}.amaxList".format(i)] = torch.tensor(
            amaxList, dtype=torch.float32
        )
    logger.info("Quantizing checkpoint done.")
    return init_dict
