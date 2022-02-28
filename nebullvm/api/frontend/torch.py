import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import numpy as np
import torch

from nebullvm.base import DeepLearningFramework, ModelParams, ModelCompiler
from nebullvm.converters import ONNXConverter
from nebullvm.converters.torch_converters import get_outputs_sizes_torch
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import ApacheTVMOptimizer
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def optimize_torch_model(
    model: torch.nn.Module,
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    save_dir: str,
    use_torch_api: bool = False,
) -> BaseInferenceLearner:
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
        use_torch_api (bool): Parameter for using the torch api of compilers
            when available. The actual implementation supports only the torch
            interface for TVM. Note that when running the torch interface
            nebullvm will ignore the ONNX one once the torch implementation
            succeeds. Clearly, in case of failure of the torch API, a second
            tentative will be done with the ONNX interface.

    Returns:
        BaseInferenceLearner: Optimized model usable with the classical Pytorch
            interface. Note that as a torch model it takes as input and it
            gives as output `torch.Tensor`s.
    """
    dl_library = DeepLearningFramework.PYTORCH
    model_params = ModelParams(
        batch_size=batch_size,
        input_sizes=input_sizes,
        output_sizes=get_outputs_sizes_torch(
            model,
            input_tensors=[
                torch.randn((batch_size, *input_size))
                for input_size in input_sizes
            ],
        ),
    )
    ignore_compilers = []
    with TemporaryDirectory() as tmp_dir:
        input_sizes = [
            (model_params.batch_size, *input_size)
            for input_size in model_params.input_sizes
        ]
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
        )
        if model_optimizer.usable:
            onnx_path = model_converter.convert(
                model, input_sizes, Path(tmp_dir)
            )
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
) -> Tuple[BaseInferenceLearner, float, List]:
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
    new_model: BaseInferenceLearner,
    previous_best_model: BaseInferenceLearner,
    previous_latency: float,
) -> BaseInferenceLearner:
    if new_model is not None:
        new_latency = compute_optimized_running_time(new_model)
        if new_latency < previous_latency:
            return new_model
    return previous_best_model
