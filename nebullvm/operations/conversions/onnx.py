import logging
from pathlib import Path

from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import Device

from onnx2torch import convert

logger = logging.getLogger("nebullvm_logger")

def convert_onnx_to_torch(
    onnx_model, # Determine datatype of onnx_model
    output_file_path: Path,
):
    """Function importing a custom ONNX model and converting it in Pytorch

    Args:
        onnx_model: ONNX model (tested with model=onnx.load("model.onnx")).
        output_file_path (str or Path): Path where storing the output 
            Pytorch file.
    """
    try:
        torch_model = convert(onnx_model)
        torch.save(torch_model, output_file_path)
        return output_file_path
    except Exception as e:
        logger.warning("Exception raised during conversion of ONNX to Pytorch."
                        "ONNX to Torch pipeline will be skipped")
        logger.warning(e)
        return None

