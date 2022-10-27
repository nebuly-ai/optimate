import logging

logger = logging.getLogger(__name__)

try:
    from deepsparse import compile_model, cpu
except ImportError:
    logger.warn(
        "deepsparse module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    compile_model = cpu = object
