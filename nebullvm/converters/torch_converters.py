from pathlib import Path
from typing import Tuple, Union, List

import torch
from torch.nn import Module

from nebullvm.base import ModelParams, DataType


def get_outputs_sizes_torch(
    torch_model: Module, input_tensors: List[torch.Tensor]
) -> List[Tuple[int, ...]]:
    if torch.cuda.is_available():
        input_tensors = [x.cuda() for x in input_tensors]
        torch_model.cuda()
    with torch.no_grad():
        outputs = torch_model(*input_tensors)
        if isinstance(outputs, torch.Tensor):
            return [tuple(outputs.size())[1:]]
        else:
            return [tuple(output.size())[1:] for output in outputs]


def convert_torch_to_onnx(
    torch_model: Module,
    model_params: ModelParams,
    output_file_path: Union[str, Path],
):
    """Function importing a custom model in pytorch and converting it in ONNX

    Args:
        torch_model (Module): Pytorch model.
        model_params (ModelParams): Model Parameters as input sizes and
            dynamic axis information.
        output_file_path (str or Path): Path where storing the output
            ONNX file.
    """
    input_tensors = [
        torch.randn(
            *(model_params.batch_size, *input_info.size), requires_grad=True
        )
        if input_info.dtype is DataType.FLOAT
        else torch.randint(
            low=input_info.min_value or 0,
            high=input_info.max_value or 100,
            size=(model_params.batch_size, *input_info.size),
        )
        for input_info in model_params.input_infos
    ]
    output_sizes = get_outputs_sizes_torch(torch_model, input_tensors)
    if torch.cuda.is_available():  # move back tensors to cpu for conversion
        input_tensors = [x.cpu() for x in input_tensors]
        torch_model.cpu()
    input_names = [f"input_{i}" for i in range(len(input_tensors))]
    output_names = [f"output_{i}" for i in range(len(output_sizes))]
    dynamic_info = model_params.dynamic_info
    if dynamic_info is not None:
        dynamic_info = {
            name: dynamic_dict
            for name, dynamic_dict in zip(
                input_names + output_names,
                dynamic_info.inputs + dynamic_info.outputs,
            )
        }
    torch.onnx.export(
        torch_model,  # model being run
        tuple(input_tensors),  # model input (or a tuple for multiple inputs)
        output_file_path,
        # where to save the model (can be a file or file-like object)
        export_params=True,
        # store the trained parameter weights inside the model file
        opset_version=11,
        # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=input_names,
        # the model's input names
        output_names=output_names,
        # the model's output names
        dynamic_axes=dynamic_info,
    )
