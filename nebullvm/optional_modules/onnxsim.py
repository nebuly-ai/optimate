import logging

logger = logging.getLogger(__name__)

try:
    import onnxsim
except ImportError:
    logger.warning(
        "Missing Library: "
        "onnxsim module is not installed on this platform. "
        "It's an optional requirement of tensorrt. "
        "Installing it could solve some issues with transformers. "
    )
    onnxsim = object
