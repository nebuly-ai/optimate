import logging

from nebullvm.utils.general import check_module_version

logger = logging.getLogger("nebullvm_logger")

try:
    import torch  # noqa F401
    from torch.nn import Module  # noqa F401
    from torch.utils.data import DataLoader  # noqa F401
    from torch.quantization.quantize_fx import (  # noqa F401
        prepare_fx,
        convert_fx,
    )

    if not check_module_version(torch, min_version="1.10.0"):
        logger.warning(
            "torch module version must be >= 1.10.0. "
            "Please update it if you want to use it."
        )
except ImportError:
    logger.warning(
        "Missing Library: "
        "torch module is not installed on this platform. "
        "Please install it if you want to use torch API."
    )
    torch = object
