from typing import (
    List,
    Dict,
    Sequence,
    Optional,
)

import numpy as np

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.optional_modules.huggingface import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
from nebullvm.tools.base import Device
from nebullvm.tools.huggingface import (
    get_output_structure_from_dict,
    get_output_structure_from_text,
    PyTorchTransformerWrapper,
    TensorFlowTransformerWrapper,
)
from nebullvm.tools.utils import is_dict_type


class _HFTextDataset(Sequence):
    def __init__(
        self,
        input_texts: List,
        ys: Optional[List],
        keywords: List[str],
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        tokenizer_args: Dict,
    ):
        self._input_texts = input_texts
        self._ys = ys
        self._bs = batch_size
        self._keys = keywords
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        _tokenizer_args = {"truncation": True, "padding": True}
        _tokenizer_args.update(tokenizer_args)
        self._tokenizer_args = _tokenizer_args

    def __getitem__(self, item: int):
        pointer = self._bs * item
        if pointer >= len(self._input_texts):
            raise IndexError
        mini_batch = self._input_texts[
            pointer : pointer + self._bs  # noqa E203
        ]
        if self._ys is not None:
            mini_batch_y = self._ys[pointer : pointer + self._bs]  # noqa E203
        else:
            mini_batch_y = None
        encoded_inputs = self._tokenizer(mini_batch, **self._tokenizer_args)
        return tuple(encoded_inputs[key] for key in self._keys), mini_batch_y

    def __len__(self):
        return len(self._input_texts) // self._bs


class _HFDictDataset(Sequence):
    def __init__(
        self,
        input_data: List,
        ys: Optional[List],
        keywords: List[str],
    ):
        self._input_data = input_data
        self._ys = ys
        self._keys = keywords

    def __getitem__(self, item: int):
        pointer = item
        if pointer >= len(self._input_data):
            raise IndexError
        mini_batch = self._input_data[pointer]
        if self._ys is not None:
            mini_batch_y = self._ys[pointer]
        else:
            mini_batch_y = None
        return (
            tuple(self._concatenate(mini_batch, key) for key in self._keys),
            mini_batch_y,
        )

    def __len__(self):
        return len(self._input_data)

    @staticmethod
    def _concatenate(mini_batch, key):
        if isinstance(mini_batch[key], torch.Tensor):
            return torch.concat([mini_batch[key]])
        elif isinstance(mini_batch[key], tf.Tensor):
            return tf.concat([mini_batch[key]], 0)
        else:
            return np.concatenate([mini_batch[key]])


def convert_hf_model(
    model: PreTrainedModel,
    input_data: List,
    device: Device,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    tokenizer_args: Optional[Dict] = None,
    batch_size: int = 1,
    **kwargs,
):
    if is_dict_type(input_data[0]):
        # already tokenized data
        if "labels" in input_data[0]:
            labels = [data.pop("labels") for data in input_data]
        else:
            labels = None
        input_example = input_data[0]
        output_structure, output_type = get_output_structure_from_dict(
            input_example=input_example,
            model=model,
            device=device,
        )
        input_data = _HFDictDataset(
            input_data=input_data,
            ys=labels,
            keywords=list(input_example.keys()),
        )

    else:
        assert tokenizer is not None, (
            "Tokenizer is needed when passing data in string format. Please "
            "provide the tokenizer as keyword argument."
        )
        if tokenizer_args is None:
            tokenizer_args = {}
        if not isinstance(input_data[0], str):
            ys = [data[1] for data in input_data]
            input_data = [data[0] for data in input_data]
        else:
            ys = None
        output_structure, output_type = get_output_structure_from_text(
            text=input_data[0],
            model=model,
            tokenizer=tokenizer,
            tokenizer_args=tokenizer_args,
            device=device,
        )
        input_example = tokenizer(input_data, **tokenizer_args)
        input_data = _HFTextDataset(
            input_texts=input_data,
            ys=ys,
            keywords=list(input_example.keys()),
            batch_size=batch_size,
            tokenizer=tokenizer,
            tokenizer_args=tokenizer_args,
        )
    if isinstance(model, torch.nn.Module):
        wrapper_model = PyTorchTransformerWrapper(
            core_model=model, encoded_input=input_example
        )
    else:
        wrapper_model = TensorFlowTransformerWrapper(
            core_model=model, encoded_input=input_example
        )

    return (
        wrapper_model,
        input_data,
        list(wrapper_model.inputs_types.keys()),
        output_structure,
        output_type,
    )
