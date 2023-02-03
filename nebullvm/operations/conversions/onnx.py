import logging
from pathlib import Path

from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import Device

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
    try:
        torch_model = convert(onnx_model)
        torch.save(torch_model, output_file_path)
        return output_file_path
    except Exception as e:
        logger.warning("Error while converting ONNX model to Pytorch")
        logger.warning(e)
        return None

