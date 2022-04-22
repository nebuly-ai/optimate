import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict

import numpy as np
import torch

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    ModelCompiler,
    InputInfo,
    QuantizationType,
)
from nebullvm.converters import ONNXConverter
from nebullvm.quantizers.onnx_quantizer import ONNXQuantizerManager
from nebullvm.utils.torch import (
    get_outputs_sizes_torch,
    create_model_inputs_torch,
)
from nebullvm.inference_learners.base import PytorchBaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import ApacheTVMOptimizer, BaseOptimizer
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def optimize_torch_model(
    model: torch.nn.Module,
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    save_dir: str,
    input_types: List[str] = None,
    extra_input_info: List[Dict] = None,
    use_torch_api: bool = False,
    dynamic_axis: Dict = None,
    quantization_ths: float = None,
    ignore_compilers: List[str] = None,
    custom_optimizers: List[BaseOptimizer] = None,
) -> PytorchBaseInferenceLearner:
    """Basic function for optimizing a torch model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.

    Args:
        model (torch.nn.Module): Pytorch model that needs optimization.
        batch_size (int): The model batch size. Note that nebullvm does not
            support at the moment dynamic batch size, so a valid input should
            be given.
        input_sizes (List[Tuple]]): List containing the size of all the input
            tensors of the model. Note that even just a single tensor is needed
            as model input, this field must be a list containing (in the
            exposed case) a single element). The tuple must contain all the
            input tensor dimensions excluding the batch size. This means that
            the final input tensor size will be considered as
            `(batch_size, *input_tensor_size)`, where `input_tensor_size` is
            one list element of `input_sizes`.
        save_dir (str): Path to the directory where saving the final model.
        input_types (List[str], optional): List of input types. If no value is
            given all the inputs will be considered as float type. The
            supported string values are "int" and "float".
        extra_input_info (List[Dict], optional): List of extra information
            needed for defining the input tensors, e.g. max_value and min_value
            the tensors can get.
        use_torch_api (bool): Parameter for using the torch api of compilers
            when available. The actual implementation supports only the torch
            interface for TVM. Note that when running the torch interface
            nebullvm will ignore the ONNX one once the torch implementation
            succeeds. Clearly, in case of failure of the torch API, a second
            tentative will be done with the ONNX interface.
        dynamic_axis (Dict, optional): Dictionary containing info about the
            dynamic axis. It should contain as keys both "inputs" and "outputs"
            and as values two lists of dictionaries where each dictionary
            represents the dynamic axis information for an input/output tensor.
            The inner dictionary should have as key an integer, i.e. the
            dynamic axis (considering also the batch size) and as value a
            string giving a "tag" to it, e.g. "batch_size".
        quantization_ths (float, optional): Tolerated relative error for
            performing quantization before compiling the model. If no value
            is given, no quantization will be performed. Note that it will not
            be used for compilers using the torch API when `use_torch_api`
            is `True`. Just dynamic quantization will be performed, since no
            data is given as input. For using other types of quantization
            please use `optimize_torch_model_from_data` instead.
        ignore_compilers (List[str], optional): List of DL compilers we want
            to ignore while running the optimization. Compiler name should be
            one between "tvm", "tensor RT", "openvino" and "onnxruntime".
        custom_optimizers (List[BaseOptimizer], optional): List of optimizers
            which can be used for producing InferenceLearners, i.e. models
            optimized for inference. This list is useful when some compilers,
            not included in the base version of nebullvm or not used
            by default, which are specific for a particular tasks, need
            to be used.

    Returns:
        PytorchBaseInferenceLearner: Optimized model usable with the classical
            Pytorch interface. Note that as a torch model it takes as input
            and it gives as output `torch.Tensor`s.
    """
    if input_types is None:
        input_types = ["float"] * len(input_sizes)
    if extra_input_info is None:
        extra_input_info = [{}] * len(input_sizes)
    if not len(input_sizes) == len(input_types) == len(extra_input_info):
        raise ValueError(
            f"Mismatch in the input list lengths. Given {len(input_sizes)} "
            f"sizes, {len(input_types)} input types and "
            f"{len(extra_input_info)} extra input infos."
        )
    input_infos = [
        InputInfo(size=input_size, dtype=input_type, **extra_info)
        for input_size, input_type, extra_info in zip(
            input_sizes, input_types, extra_input_info
        )
    ]
    dl_library = DeepLearningFramework.PYTORCH
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=input_infos,
        output_sizes=get_outputs_sizes_torch(
            model,
            input_tensors=create_model_inputs_torch(batch_size, input_infos),
        ),
        dynamic_info=dynamic_axis,
    )
    ignore_compilers = (
        []
        if ignore_compilers is None
        else [ModelCompiler(compiler) for compiler in ignore_compilers]
    )
    with TemporaryDirectory() as tmp_dir:
        if use_torch_api:
            (
                torch_api_model,
                torch_api_latency,
                used_compilers,
            ) = _torch_api_optimization(model, model_params)
            ignore_compilers.extend(used_compilers)
        model_converter = ONNXConverter()
        model_optimizer = MultiCompilerOptimizer(
            ignore_compilers=ignore_compilers,
            extra_optimizers=custom_optimizers,
            debug_mode=int(os.environ.get("DEBUG_MODE", "0")) > 0,
        )
        if model_optimizer.usable:
            onnx_path = model_converter.convert(
                model, model_params, Path(tmp_dir)
            )
            if quantization_ths is not None:
                quantization_manager = ONNXQuantizerManager(quantization_ths)
                quantized_onnx_path = quantization_manager.run(
                    str(onnx_path),
                    model_params,
                    quantization_type=QuantizationType.DYNAMIC,
                )
                if quantized_onnx_path is not None:
                    onnx_path = Path(quantized_onnx_path)
            model_optimized = model_optimizer.optimize(
                str(onnx_path), dl_library, model_params
            )
        else:
            model_optimized = None
        if model_optimized is None and not ignore_compilers:
            raise RuntimeError(
                "No valid compiled model has been produced. "
                "Look at the logs for further information about the failure."
            )
        if use_torch_api:
            model_optimized = _compare_optimized_models(
                model_optimized,
                torch_api_model,
                torch_api_latency,
            )
        model_optimized.save(save_dir)
    return model_optimized.load(save_dir)


def _torch_api_optimization(
    model: torch.nn.Module, model_params: ModelParams
) -> Tuple[PytorchBaseInferenceLearner, float, List]:
    try:
        best_torch_opt_model = ApacheTVMOptimizer().optimize_from_torch(
            torch_model=model, model_params=model_params
        )
        best_latency = compute_optimized_running_time(best_torch_opt_model)
        used_compilers = [ModelCompiler.APACHE_TVM]
    except Exception as ex:
        warnings.warn(
            f"Compilation failed with torch interface of TVM. "
            f"Got error {ex}. The compilation will be re-scheduled "
            f"with the ONNX interface."
        )
        best_torch_opt_model = None
        best_latency = np.inf
        used_compilers = []
    return best_torch_opt_model, best_latency, used_compilers


def _compare_optimized_models(
    new_model: PytorchBaseInferenceLearner,
    previous_best_model: PytorchBaseInferenceLearner,
    previous_latency: float,
) -> PytorchBaseInferenceLearner:
    if new_model is not None:
        new_latency = compute_optimized_running_time(new_model)
        if new_latency < previous_latency:
            return new_model
    return previous_best_model
