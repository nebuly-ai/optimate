from pathlib import Path
from typing import Tuple, Union

import torch
from torch.nn import Module


def convert_torch_to_onnx(
    torch_model: Module,
    input_size: Tuple[int, ...],
    output_file_path: Union[str, Path],
):
    """Function importing a custom model in pytorch and converting it in ONNX

    Parameters
    ----------
        torch_model (Module): Pytorch model.
        input_size (Tuple[int]): Size of the input tensor.
            Should contain the batchsize as well.
        output_file_path (str or Path): Path where storing the output
            ONNX file.
    """
    x = torch.randn(*input_size, requires_grad=True)
    if torch.cuda.is_available():  # move back tensors to cpu for conversion
        x = x.cpu()
        torch_model.cpu()
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        output_file_path,
        # where to save the model (can be a file or file-like object)
        export_params=True,
        # store the trained parameter weights inside the model file
        opset_version=11,
        # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     # variable length axes
        #     "output": {0: "batch_size"},
        # },
    )
