from pathlib import Path

from loguru import logger

from nebullvm.config import ONNX_OPSET_VERSION
from nebullvm.optional_modules.torch import torch, Module
from nebullvm.tools.base import ModelParams, Device, DeviceType
from nebullvm.tools.data import DataManager
from nebullvm.tools.pytorch import (
    get_outputs_sizes_torch,
    create_model_inputs_torch,
)


def convert_torch_to_onnx(
    torch_model: Module,
    input_data: DataManager,
    model_params: ModelParams,
    output_file_path: Path,
    device: Device,
):
    """Function importing a custom model in pytorch and converting it in ONNX

    Args:
        torch_model (Module): Pytorch model.
        input_data (DataManager): Custom data provided by user to be
        used as input for the converter.
        model_params (ModelParams): Model Parameters as input sizes and
            dynamic axis information.
        output_file_path (str or Path): Path where storing the output
            ONNX file.
        device (Device): Device where the model will be run.
    """

    if input_data is not None:
        input_tensors = list(input_data.get_list(1)[0])
    else:
        input_tensors = create_model_inputs_torch(model_params.input_infos)

    output_sizes = get_outputs_sizes_torch(torch_model, input_tensors, device)

    input_names = [f"input_{i}" for i in range(len(input_tensors))]
    output_names = [f"output_{i}" for i in range(len(output_sizes))]
    dynamic_info = model_params.dynamic_info

    if dynamic_info is not None:
        # This check is needed to enable backward compatibility with
        # previous versions of nebullvm
        if isinstance(list(dynamic_info.inputs[0].values())[0], str):
            onnx_format_inputs = dynamic_info.inputs
        else:
            onnx_format_inputs = [
                {k: v["name"] for (k, v) in d.items()}
                for d in dynamic_info.inputs
            ]

        assert len(dynamic_info.outputs) == len(output_names), (
            f"The number of dynamic outputs provided in the dynamic info "
            f"dict ({len(dynamic_info.outputs)}) is not equal to the number "
            f"of outputs of the model ({len(output_names)}), Detected model "
            f"output shapes are: {output_sizes} "
        )

        dynamic_info = {
            name: dynamic_dict
            for name, dynamic_dict in zip(
                input_names + output_names,
                onnx_format_inputs + dynamic_info.outputs,
            )
        }

    try:
        # try conversion with model on cpu
        if device.type is DeviceType.GPU:
            input_tensors = [x.cpu() for x in input_tensors]
            torch_model.cpu()

        torch.onnx.export(
            torch_model,  # model being run
            tuple(
                input_tensors
            ),  # model input (or a tuple for multiple inputs)
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

        # Put again model on gpu
        if device.type is DeviceType.GPU:
            torch_model.to(device.to_torch_format())

        return output_file_path
    except Exception:
        # try conversion with model on gpu
        if device.type is DeviceType.GPU:
            input_tensors = [
                x.to(device.to_torch_format()) for x in input_tensors
            ]
            torch_model.to(device.to_torch_format())

            try:
                torch.onnx.export(
                    torch_model,  # model being run
                    tuple(
                        input_tensors
                    ),  # model input (or a tuple for multiple inputs)
                    str(output_file_path),
                    # where to save the model
                    # (can be a file or file-like object)
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

                return output_file_path
            except Exception:
                logger.warning(
                    "Exception raised during conversion from torch"
                    " to onnx model. ONNX pipeline will be unavailable."
                )
                return None
        else:
            logger.warning(
                "Exception raised during conversion from torch"
                " to onnx model. ONNX pipeline will be unavailable."
            )
            return None
