import logging
from pathlib import Path

from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import Device
from nebullvm.optional_modules.onnx import ModelProto
logger = logging.getLogger("nebullvm_logger")

from nebullvm.optional_modules.onnx import convert

def convert_onnx_to_torch(
    onnx_model: ModelProto
):
    """Function importing a custom ONNX model and converting it in Pytorch

    Args:
        onnx_model: ONNX model (tested with model=onnx.load("model.onnx")).
    """
    try:
        torch_model = torch.fx.symbolic_trace(convert(onnx_model))
        return torch_model
    except Exception as e:
        logger.warning("Exception raised during conversion of ONNX to Pytorch."
                        "ONNX to Torch pipeline will be skipped")
        logger.warning(e)
        return None

