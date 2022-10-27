import logging

from nebullvm.utils.general import check_module_version

logger = logging.getLogger(__name__)

try:
    import onnx  # noqa F401

    if not check_module_version(onnx, min_version="1.10.0"):
        logger.warning(
            "onnx module version must be >= 1.10.0. "
            "Please update it if you want to use it."
        )
except ImportError:
    logger.warning(
        "onnx module is not installed on this platform. "
        "Please install it if you want to use ONNX API or "
        "the ONNX pipeline for PyTorch and Tensorflow."
    )
    onnx = None

try:
    import onnxmltools  # noqa F401
    from onnxmltools.utils.float16_converter import (  # noqa F401
        convert_float_to_float16_model_path,
    )

    if not check_module_version(onnxmltools, min_version="1.11.0"):
        logger.warning(
            "onnxmltools module version must be >= 1.11.0. "
            "Please update it if you want to use the ONNX API "
            "or the ONNX pipeline for PyTorch and Tensorflow."
        )

except ImportError:
    logger.warning(
        "onnxmltools module is not installed on this platform. "
        "Please install it if you want to use ONNX API or "
        "the ONNX pipeline for PyTorch and Tensorflow."
    )
    convert_float_to_float16_model_path = object
