from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import (
    Tuple,
    Union,
    List,
    Iterable,
    Dict,
    Any,
    Type,
    Callable,
    Optional,
    Sequence,
)

import numpy as np
import torch

from nebullvm import optimize_torch_model
from nebullvm.api.frontend.utils import ifnone, QUANTIZATION_METRIC_MAP
from nebullvm.base import DataType, ModelCompiler
from nebullvm.inference_learners.base import (
    PytorchBaseInferenceLearner,
    InferenceLearnerWrapper,
    LearnerMetadata,
)
from nebullvm.optimizers.extra import HuggingFaceOptimizer

try:
    from transformers import PreTrainedModel, PretrainedConfig
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


def _get_output_structure(
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
            output_dict[key] = (
                np.array(
                    output[idx : int(np.prod(value)) + idx],  # noqa E203
                    dtype=object,
                )
                .reshape(value)
                .tolist()
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


def _get_dynamic_axis(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    tokenizer_args: Dict,
) -> Dict[str, List[Dict[int, str]]]:
    input_1 = tokenizer([text], **tokenizer_args)
    input_2 = tokenizer([text + text], **tokenizer_args)
    input_dicts = []
    for key in input_1.keys():
        input_dict = {}
        for idx, (i, j) in enumerate(
            zip(input_1[key].shape, input_2[key].shape)
        ):
            if i != j:
                input_dict[idx] = f"val_{i}_{j}"
        input_dicts.append(input_dict)

    output_dicts = []
    outputs_1 = _flatten_outputs(model(**input_1).values())
    outputs_2 = _flatten_outputs(model(**input_2).values())
    for o1, o2 in zip(outputs_1, outputs_2):
        output_dict = {}
        for idx, (i, j) in enumerate(zip(o1.shape, o2.shape)):
            if i != j:
                output_dict[idx] = f"val_{i}_{j}"
        output_dicts.append(output_dict)
    return {"inputs": input_dicts, "outputs": output_dicts}


def _extract_input_type(input_value: torch.Tensor):
    if input_value.dtype is torch.float:
        return DataType.FLOAT
    elif input_value.dtype is torch.long:
        return DataType.INT
    else:
        raise NotImplementedError(
            f"Unsupported data format {input_value.dtype}."
        )


def _try_extraction(model_config: PretrainedConfig, keys: List[str]):
    for key in keys:
        if hasattr(model_config, key):
            return getattr(model_config, key)
    return


def _get_extra_optimizer(
    model_config: PretrainedConfig,
) -> List[HuggingFaceOptimizer]:
    config_name = model_config.__class__.__name__.lower()
    for key in HuggingFaceOptimizer.get_accepted_types():
        if key in config_name:
            input_dict = {"model_type": key, "opt_level": 2}
            hidden_dim = _try_extraction(
                model_config, ["n_embd", "d_model", "hidden_size"]
            )
            if hidden_dim is not None:
                input_dict["hidden_size"] = hidden_dim
            n_heads = _try_extraction(
                model_config,
                ["n_head", "num_attention_heads", "encoder_attention_heads"],
            )
            if n_heads is not None:
                input_dict["num_heads"] = n_heads
            return [HuggingFaceOptimizer(hugging_face_params=input_dict)]
    return [HuggingFaceOptimizer(hugging_face_params={})]


class _HFDataset(Sequence):
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
        if pointer >= len(self):
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
        return len(self._input_texts)


def optimize_huggingface_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_texts: List[str],
    batch_size: int,
    max_input_sizes: List[Tuple[int, ...]],
    save_dir: str,
    extra_input_info: List[Dict] = None,
    use_static_shape: bool = False,
    use_torch_api: bool = False,
    tokenizer_args: Dict = None,
    ignore_compilers: List[str] = None,
    perf_loss_ths: float = None,
    perf_metric: Union[str, Callable] = None,
    ys: List = None,
):
    """Optimize the HuggingFace model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.
    Note that TensorRT compiler is currently disabled for Hugginface models
    since in some cases it can cause an untreatable error in the C++ code
    causing the interruption of the optimization.

    Args:
        model (PreTrainedModel): HuggingFace transformers model.
        tokenizer (PreTrainedTokenizer): Tokenizer used for building model's
            inputs.
        input_texts (List[str]): Texts either from the training set or similar
            to the ones contained in the text. If the perf_loss_ths is
            passed the input_text will be used for computing the drop in
            precision and for setting the quantization parameters. If you
            selected a quantization metric needing the input labels you need to
            provide them for each input in the `ys` argument.
        batch_size (int): Batch size needed for the model.
        max_input_sizes (List[Tuple[int]]): List containing the maximum size of
            all the input tensors of the model.
            Note that even just a single tensor is needed as model input,
            this field must be a list containing (in the exposed case)
            a single element). The tuple must contain the maximum value for
            all the input tensor dimensions excluding the batch size.
            This means that the final input tensor size will be considered as
            `(batch_size, *input_tensor_size)`, where `input_tensor_size` is
            one list element of `max_input_sizes`.
        save_dir (str):  Path to the directory where saving the final model.
        extra_input_info (List[Dict], optional): List of extra information
            needed for defining the input tensors, e.g. max_value and min_value
            the tensors can get.
        use_static_shape (bool): Parameter for fixing the accepted input shape.
        use_torch_api (bool): Parameter for using the torch api of compilers
            when available. The actual implementation supports only the torch
            interface for TVM. Note that when running the torch interface
            nebullvm will ignore the ONNX one once the torch implementation
            succeeds. Clearly, in case of failure of the torch API, a second
            tentative will be done with the ONNX interface.
        tokenizer_args (Dict, optional): Extra args needed for the tokenizer.
        ignore_compilers (List[str], optional): List of DL compilers we want
            to ignore while running the optimization. Compiler name should be
            one between "tvm", "tensor RT", "openvino" and "onnxruntime".
        perf_loss_ths (float, optional): Tolerated relative error for
            performing approximation techniques before compiling the model.
            If no value is given, no optimization will be performed. Note that
            it will not be used for compilers using the torch API when
            `use_torch_api` is `True`. Just dynamic quantization will be
            performed, since no data is given as input.
        perf_metric (Union[Callable, str], optional): The metric to
            be used for accepting or refusing a precision-reduction
            optimization proposal. If none is given but a `perf_loss_ths` is
            received, the `nebullvm.measure.compute_relative_difference`
            metric will be used as default one. A user-defined metric can
            be passed as function accepting as inputs two tuples of tensors
            (produced by the baseline and the quantized model) and the related
            original labels.
            For more information see
            `nebullvm.measure.compute_relative_difference` and
            `nebullvm.measure.compute_accuracy_drop`. `perf_metric`
            accepts as value also a string containing the metric name. At the
            current stage the supported metrics are `"precision"` and
            `"accuracy"`.
        ys: List of target labels. For each input in `input_texts` there should
            be the corresponding label. Note that this feature is just used for
            estimating the accuracy drop while running precision-reduction
            techniques. It will be ignored if these techniques are not
            activated.
    """
    if perf_loss_ths is not None and ys is None and perf_metric == "accuracy":
        raise ValueError(
            "You cannot select the accuracy as quantization metric without "
            "providing valid labels!"
        )
    if isinstance(perf_metric, str):
        perf_metric = QUANTIZATION_METRIC_MAP.get(perf_metric)
    tokenizer_args = tokenizer_args or {}
    tokenizer_args.update({"return_tensors": "pt"})
    output_structure, output_type = _get_output_structure(
        text=input_texts[0],
        model=model,
        tokenizer=tokenizer,
        tokenizer_args=tokenizer_args,
    )
    input_example = tokenizer(input_texts[0], **tokenizer_args)
    input_types = [_extract_input_type(v) for v in input_example.values()] or [
        "int"
    ] * len(input_example)
    # The wrapper model is needed for adapt the huggingface transformers API
    # to the one adopted by the nebullvm optimization.
    wrapper_model = _TransformerWrapper(
        core_model=model, encoded_input=input_example
    )
    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_torch_model(
            wrapper_model,
            batch_size=batch_size,
            input_sizes=max_input_sizes,
            save_dir=tmp_dir,
            input_types=input_types,
            extra_input_info=extra_input_info,
            use_torch_api=use_torch_api,
            dynamic_axis=_get_dynamic_axis(
                text=input_texts[0],
                tokenizer=tokenizer,
                model=model,
                tokenizer_args=tokenizer_args,
            )
            if not use_static_shape
            else None,
            perf_loss_ths=perf_loss_ths,
            perf_metric=perf_metric,
            dataloader=_HFDataset(
                input_texts,
                ys,
                list(wrapper_model.inputs_types.keys()),
                batch_size,
                tokenizer,
                tokenizer_args,
            ),
            ignore_compilers=list(
                set(
                    (
                        [ModelCompiler.TENSOR_RT.value]
                        if use_static_shape
                        else [
                            ModelCompiler.TENSOR_RT.value,
                            ModelCompiler.APACHE_TVM.value,
                        ]
                    )
                    + ifnone(ignore_compilers, [])
                )
            ),
            custom_optimizers=_get_extra_optimizer(model.config),
        )
        final_model = HuggingFaceInferenceLearner(
            core_inference_learner=optimized_model,
            output_structure=output_structure,
            input_names=list(wrapper_model.inputs_types.keys()),
            output_type=output_type,
        )
        final_model.save(save_dir)

    return final_model.load(save_dir)
