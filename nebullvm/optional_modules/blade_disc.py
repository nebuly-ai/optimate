import logging

logger = logging.getLogger("nebullvm_logger")

try:
    import torch_blade
except ImportError:
    logger.warning(
        "Missing Library: "
        "torch_blade module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )

    torch_blade = object