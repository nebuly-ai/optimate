import logging
from pathlib import Path

from nebullvm.config import ONNX_OPSET_VERSION
from nebullvm.optional_modules.torch import torch, Module
from nebullvm.tools.base import ModelParams, Device
from nebullvm.tools.data import DataManager
from nebullvm.tools.pytorch import (
    get_outputs_sizes_torch,
    create_model_inputs_torch,
)
from onnx2torch import convert

logger = logging.getLogger("nebullvm_logger")

def convert_onnx_to_torch(
    onnx_model, # Determine datatype of onnx_model
    output_file_path: Path,
    device: Device,
):
    """Function importing a custom ONNX model and converting it in Pytorch

    Args:
        onnx_model: ONNX model (tested with model=onnx.load("model.onnx")).
        output_file_path (str or Path): Path where storing the output 
            Pytorch file.
        device (Device): Device where the model will be run.
        
    """
    torch_model = convert(onnx_model)
    torch.save(torch_model, output_file_path)

    return output_file_path
            