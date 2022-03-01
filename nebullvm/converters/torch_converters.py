from pathlib import Path
from typing import Tuple, Union, List

import torch
from torch.nn import Module


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
    input_sizes: List[Tuple[int, ...]],
    output_file_path: Union[str, Path],
):
    """Function importing a custom model in pytorch and converting it in ONNX

    Args:
        torch_model (Module): Pytorch model.
        input_sizes (List[Tuple[int]]): Sizes of the input tensors.
            Should contain the batch size as well.
        output_file_path (str or Path): Path where storing the output
            ONNX file.
    """
    input_tensors = [
        torch.randn(*input_size, requires_grad=True)
        for input_size in input_sizes
    ]
    output_sizes = get_outputs_sizes_torch(torch_model, input_tensors)
    if torch.cuda.is_available():  # move back tensors to cpu for conversion
        input_tensors = [x.cpu() for x in input_tensors]
        torch_model.cpu()
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
        input_names=[f"input_{i}" for i in range(len(input_tensors))],
        # the model's input names
        output_names=[f"output_{i}" for i in range(len(output_sizes))],
        # the model's output names
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     # variable length axes
        #     "output": {0: "batch_size"},
        # },
    )
