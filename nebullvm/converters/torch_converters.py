from pathlib import Path
from typing import Union

import torch
from torch.nn import Module

from nebullvm.base import ModelParams
from nebullvm.config import ONNX_OPSET_VERSION
from nebullvm.utils.data import DataManager
from nebullvm.utils.torch import (
    get_outputs_sizes_torch,
    create_model_inputs_torch,
)


def convert_torch_to_onnx(
    torch_model: Module,
    model_params: ModelParams,
    output_file_path: Union[str, Path],
    input_data: DataManager = None,
):
    """Function importing a custom model in pytorch and converting it in ONNX

    Args:
        torch_model (Module): Pytorch model.
        model_params (ModelParams): Model Parameters as input sizes and
            dynamic axis information.
        output_file_path (str or Path): Path where storing the output
            ONNX file.
        input_data (DataManager, optional): Custom data provided by user to be
        used as input for the converter.
    """

    if input_data is not None:
        input_tensors = list(input_data.get_list(1)[0])
    else:
        input_tensors = create_model_inputs_torch(
            model_params.batch_size, model_params.input_infos
        )

    output_sizes = get_outputs_sizes_torch(torch_model, input_tensors)
    if torch.cuda.is_available():  # move tensors to gpu if cuda is available
        input_tensors = [x.cuda() for x in input_tensors]
        torch_model.cuda()
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
        str(output_file_path),
        # where to save the model (can be a file or file-like object)
        export_params=True,
        # store the trained parameter weights inside the model file
        opset_version=ONNX_OPSET_VERSION,
        # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=input_names,
        # the model's input names
        output_names=output_names,
        # the model's output names
        dynamic_axes=dynamic_info,
    )
