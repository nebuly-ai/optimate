import logging

logger = logging.getLogger(__name__)

try:
    import torch_tensorrt
except ImportError:
    logger.warning(
        "Missing Library: "
        "torch_tensorrt module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    torch_tensorrt = object
