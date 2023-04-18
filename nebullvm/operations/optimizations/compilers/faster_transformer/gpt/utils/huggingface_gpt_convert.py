# Based on https://github.com/NVIDIA/FasterTransformer/blob/4402759e48f2340220638675f464b6ba1f79ac3c/examples/pytorch/gpt/utils/huggingface_gpt_convert.py # noqa: E501
# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
Convert huggingface GPT model. Use https://huggingface.co/gpt2 as demo.
"""

import argparse
import configparser
import os
import sys

from loguru import logger
import numpy as np
from transformers import GPT2Model  # transformers-4.10.0-py3

from nebullvm.optional_modules.torch import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(i, saved_dir, factor, key, args, val):

    if (
        key.find("input_layernorm.weight") != -1
        or key.find("input_layernorm.bias") != -1
        or key.find("attention.dense.bias") != -1
        or key.find("post_attention_layernorm.weight") != -1
        or key.find("post_attention_layernorm.bias") != -1
        or key.find("mlp.dense_4h_to_h.bias") != -1
        or key.find("final_layernorm.weight") != -1
        or key.find("final_layernorm.bias") != -1
    ):

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir + "/model." + key + ".bin"
            val.tofile(saved_path)

    elif (
        key.find("attention.dense.weight") != -1
        or key.find("mlp.dense_4h_to_h.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = (
                saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            )
            split_vals[j].tofile(saved_path)

    elif (
        key.find("mlp.dense_h_to_4h.weight") != -1
        or key.find("mlp.dense_h_to_4h.bias") != -1
    ):

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = (
                saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            )
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        local_dim = (int)(val.shape[-1] / 3)

        val = val.reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = (
                saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            )
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        hidden_dim = val.shape[0]
        local_dim = (int)(val.shape[-1] / 3)

        val = val.reshape(hidden_dim, 3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = (
                saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            )
            split_vals[j].tofile(saved_path)

    else:
        logger.warning("[ERROR] cannot find key '{}'".format(key))


def split_and_convert(args):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2Model.from_pretrained(args.in_file).to(torch_device)
    main(
        args.saved_dir,
        model,
        args.trained_gpu_num,
        args.infer_gpu_num,
        args.processes,
        args.weight_data_type,
    )


def main(
    saved_dir,
    model: GPT2Model,
    trained_gpu_num=1,
    infer_gpu_num=1,
    processes=1,
    weight_data_type="fp32",
):
    assert isinstance(model, GPT2Model), "model must be GPT2Model"
    args = None
    saved_dir = saved_dir + "/%d-gpu/" % infer_gpu_num

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    # ckpt_name = args.in_file

    t_gpu_num = trained_gpu_num
    i_gpu_num = infer_gpu_num
    assert i_gpu_num % t_gpu_num == 0

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = GPT2Model.from_pretrained(args.in_file).to(torch_device)

    hf_config = vars(model.config)

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    config["gpt"] = {}
    try:
        config["gpt"]["model_name"] = (
            "gpt"
            if hf_config["_name_or_path"] == ""
            else hf_config["_name_or_path"]
        )
        config["gpt"]["head_num"] = str(hf_config["n_head"])
        n_embd = hf_config["n_embd"]
        config["gpt"]["size_per_head"] = str(n_embd // hf_config["n_head"])
        config["gpt"]["inter_size"] = str(n_embd * 4)
        config["gpt"]["max_pos_seq_len"] = str(hf_config["n_positions"])
        config["gpt"]["num_layer"] = str(hf_config["n_layer"])
        config["gpt"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gpt"]["start_id"] = str(hf_config["bos_token_id"])
        config["gpt"]["end_id"] = str(hf_config["eos_token_id"])
        config["gpt"]["weight_data_type"] = weight_data_type
        with open(saved_dir + "/config.ini", "w") as configfile:
            config.write(configfile)
    except:  # noqa: E722
        logger.warning("Fail to save the config in config.ini.")

    np_weight_data_type = get_weight_data_type(weight_data_type)

    huggingface_model_name_pattern = [
        "ln_1.bias",
        "ln_1.weight",
        "attn.c_attn.bias",
        "attn.c_attn.weight",
        "attn.c_proj.bias",
        "attn.c_proj.weight",
        "ln_2.bias",
        "ln_2.weight",
        "mlp.c_fc.bias",
        "mlp.c_fc.weight",
        "mlp.c_proj.bias",
        "mlp.c_proj.weight",
    ]

    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    # torch.multiprocessing.set_start_method("spawn")
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # pool = multiprocessing.Pool(args.processes)
    for name, param in model.named_parameters():
        if name.find("weight") == -1 and name.find("bias") == -1:
            continue
        if name == "wpe.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.wpe.bin"
            )
        elif name == "wte.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.wte.bin"
            )
        elif name == "ln_f.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.final_layernorm.bias.bin"
            )
        elif name == "ln_f.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.final_layernorm.weight.bin"
            )
        elif name == "lm_head.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir + "model.lm_head.weight.bin"
            )
        else:
            for i in range(len(huggingface_model_name_pattern)):
                if name.find(huggingface_model_name_pattern[i]) != -1:
                    new_name = name.replace("h.", "layers.").replace(
                        huggingface_model_name_pattern[i],
                        ft_model_name_pattern[i],
                    )
                    # pool.starmap(split_and_convert_process,
                    # [(0, saved_dir, factor, new_name, args,
                    # param.detach().cpu().numpy().astype(np_weight_data_type))],
                    # )
                    split_and_convert_process(
                        0,
                        saved_dir,
                        factor,
                        new_name,
                        args,
                        param.detach()
                        .cpu()
                        .numpy()
                        .astype(np_weight_data_type),
                    )

    # pool.close()
    # pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-saved_dir",
        "-o",
        type=str,
        help="file name of output file",
        required=True,
    )
    parser.add_argument(
        "-in_file",
        "-i",
        type=str,
        help="file name of input checkpoint file",
        required=True,
    )
    parser.add_argument(
        "-trained_gpu_num",
        "-t_g",
        type=int,
        help="How many gpus for inference",
        default=1,
    )
    parser.add_argument(
        "-infer_gpu_num",
        "-i_g",
        type=int,
        help="How many gpus for inference",
        required=True,
    )
    parser.add_argument(
        "-processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4,
    )
    parser.add_argument(
        "-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"]
    )

    args = parser.parse_args()
    logger.info("\n=============== Argument ===============")
    for key in vars(args):
        logger.info("{}: {}".format(key, vars(args)[key]))
    logger.info("========================================")

    split_and_convert(args)
