from collections import OrderedDict
from typing import (
    Union,
    Iterable,
    List,
    Dict,
    Tuple,
    Type,
    Any,
    Sequence,
    Optional,
)

import numpy as np
import torch

from nebullvm.inference_learners import (
    InferenceLearnerWrapper,
    PytorchBaseInferenceLearner,
    LearnerMetadata,
)

try:
    from transformers import (
        PreTrainedModel,
    )
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError:
    # add placeholders for function definition
    PreTrainedModel = None
    PreTrainedTokenizer = None


def _flatten_outputs(
    outputs: Union[torch.Tensor, Iterable]
) -> List[torch.Tensor]:
    new_outputs = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            new_outputs.append(output)
        else:
            flatten_list = _flatten_outputs(output)
            new_outputs.extend(flatten_list)
    return new_outputs


class _TransformerWrapper(torch.nn.Module):
    """Class for wrappering the Transformers and give them an API compatible
    with nebullvm. The class takes and input of the forward method positional
    arguments and transform them in the input dictionaries needed by
    transformers classes. At the end it also flattens their output.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        encoded_input: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.core_model = core_model
        self.inputs_types = OrderedDict()
        for key, value in encoded_input.items():
            self.inputs_types[key] = value.dtype

    def forward(self, *args: torch.Tensor):
        inputs = {
            key: value for key, value in zip(self.inputs_types.keys(), args)
        }
        outputs = self.core_model(**inputs)
        return tuple(_flatten_outputs(outputs.values()))


def _get_size_recursively(
    tensor_tuple: Union[torch.Tensor, Tuple]
) -> List[int]:
    if isinstance(tensor_tuple[0], torch.Tensor):
        return [len(tensor_tuple)]
    else:
        inner_size = _get_size_recursively(tensor_tuple[0])
        return [len(tensor_tuple), *inner_size]


def _get_output_structure_from_text(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenizer_args: Dict,
) -> Tuple[OrderedDict, Type]:
    """Function needed for saving in a dictionary the output structure of the
    transformers model.
    """
    encoded_input = tokenizer([text], **tokenizer_args)
    output = model(**encoded_input)
    structure = OrderedDict()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            structure[key] = None
        else:
            size = _get_size_recursively(value)
            structure[key] = size
    return structure, type(output)


def _get_output_structure_from_dict(
    input_example: Dict,
    model: PreTrainedModel,
) -> Tuple[OrderedDict, Type]:
    """Function needed for saving in a dictionary the output structure of the
    transformers model.
    """
    output = model(**input_example)
    structure = OrderedDict()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            structure[key] = None
        else:
            size = _get_size_recursively(value)
            structure[key] = size
    return structure, type(output)


def _restructure_output(
    output: Tuple[torch.Tensor],
    structure: OrderedDict,
    output_type: Any = None,
):
    """Restructure the flatter output using the structure dictionary given as
    input.
    """
    output_dict = {}
    idx = 0
    for key, value in structure.items():
        if value is None:
            output_dict[key] = output[idx]
            idx += 1
        else:
            tensor_shape = output[idx].shape[1:]
            output_dict[key] = list(
                torch.reshape(
                    torch.stack(
                        output[idx : int(np.prod(value)) + idx]  # noqa E203
                    ),
                    (*value, *tensor_shape),
                )
            )
            idx += np.prod(value)
    if output_type is not None:
        return output_type(**output_dict)
    return output_dict


class HuggingFaceInferenceLearner(InferenceLearnerWrapper):
    """Class wrapping an InferenceLearner model and giving to it the
    huggingface interface.

    The class fuse both the InterfaceLearner and HuggingFace interfaces, giving
    to the final user a model which can be used whit the prefered API without
    the need of adapting the previous code.

    Attributes:
        network_parameters (ModelParams): Model parameters of the model.
        core_inference_learner (PytorchBaseInferenceLearner): Inference learner
            built using the Pytorch interface.
        output_structure (Dict): Original output structure of the HuggingFace
            model.
        input_names (List[str]): List of all the input keys used for the
            original HuggingFace model.
        output_type (Any, optional): Original output type of the HuggingFace
            model.
    """

    def __init__(
        self,
        core_inference_learner: PytorchBaseInferenceLearner,
        output_structure: OrderedDict,
        input_names: List[str],
        output_type: Any = None,
    ):
        super().__init__(core_inference_learner)
        self.output_structure = output_structure
        self.input_names = input_names
        self.output_type = output_type

    def _save_wrapper_extra_info(self):
        pass

    @staticmethod
    def _load_wrapper_extra_info(builder_inputs: Dict) -> Dict:
        return builder_inputs

    def run(self, *args, **kwargs) -> Any:
        """Run the underlying optimized model for getting a prediction.

        The method has an hybrid interface. It accepts inputs either as
        positional or keyword arguments. If only positional arguments are given
        the method expects the inputs to be in the canonical
        nebullvm interface. If only keyword arguments are given the method
        expects them to be in the HuggingFace interface. Mixed representation
        is not allowed and will result in an error.
        """
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError(
                "Not allowed usage of the predict method. "
                "Either the positional or the keyword arguments must be given."
            )
        if len(args) > 0:
            return self.core_inference_learner(*args)
        inputs = (kwargs.pop(name) for name in self.input_names)
        outputs = self.core_inference_learner(*inputs)
        return _restructure_output(
            outputs, self.output_structure, self.output_type
        )

    def _get_extra_metadata_kwargs(self) -> Dict:
        metadata_kwargs = {
            "output_structure": self.output_structure,
            "output_structure_keys": list(self.output_structure.keys()),
            "input_names": self.input_names,
        }
        if self.output_type is not None:
            metadata_kwargs.update(
                {
                    "output_type": self.output_type.__name__,
                    "output_type_module": self.output_type.__module__,
                }
            )
        return metadata_kwargs

    @staticmethod
    def _convert_metadata_to_inputs(metadata: LearnerMetadata) -> Dict:
        # we need to guarantee the preservation of the output structure
        # elements order.
        output_structure = OrderedDict()
        for key in metadata["output_structure_keys"]:
            output_structure[key] = metadata["output_structure"][key]

        inputs = {
            "output_structure": output_structure,
            "input_names": metadata["input_names"],
        }
        if metadata["output_type"] is not None:
            exec(
                f"from {metadata['output_type_module']} "
                f"import {metadata['output_type']}"
            )
            inputs["output_type"] = eval(metadata["output_type"])
        return inputs


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
            tuple(torch.concat([mini_batch[key]]) for key in self._keys),
            mini_batch_y,
        )

    def __len__(self):
        return len(self._input_data)


def convert_hf_model(
    model: PreTrainedModel,
    input_data: List,
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
        output_structure, output_type = _get_output_structure_from_dict(
            input_example=input_example,
            model=model,
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
        output_structure, output_type = _get_output_structure_from_text(
            text=input_data[0],
            model=model,
            tokenizer=tokenizer,
            tokenizer_args=tokenizer_args,
        )
        input_example = tokenizer(input_data)
        input_data = _HFTextDataset(
            input_texts=input_data,
            ys=ys,
            keywords=list(input_example.keys()),
            batch_size=batch_size,
            tokenizer=tokenizer,
            tokenizer_args=tokenizer_args,
        )
    wrapper_model = _TransformerWrapper(
        core_model=model, encoded_input=input_example
    )
    return (
        wrapper_model,
        input_data,
        list(wrapper_model.inputs_types.keys()),
        output_structure,
        output_type,
    )


def is_dict_type(data_sample: Any):
    try:
        data_sample.items()
    except AttributeError:
        return False
    else:
        return True
