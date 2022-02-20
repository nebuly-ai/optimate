from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import torch

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.converters import ONNXConverter
from nebullvm.converters.torch_converters import get_outputs_sizes_torch
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def optimize_torch_model(
    model: torch.nn.Module,
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    save_dir: str,
):
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
    model_converter = ONNXConverter()
    model_optimizer = MultiCompilerOptimizer(n_jobs=-1)
    with TemporaryDirectory() as tmp_dir:
        input_sizes = [
            (model_params.batch_size, *input_size)
            for input_size in model_params.input_sizes
        ]
        onnx_path = model_converter.convert(model, input_sizes, Path(tmp_dir))
        model_optimized = model_optimizer.optimize(
            str(onnx_path), dl_library, model_params
        )
        model_optimized.save(save_dir)
    return model_optimized.load(save_dir)
