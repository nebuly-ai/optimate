import logging

logger = logging.getLogger("nebullvm_logger")

try:
    from deepsparse import compile_model, cpu
except ImportError:
    logger.warning(
        "Missing Library: "
        "deepsparse module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    compile_model = cpu = object
