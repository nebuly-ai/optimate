import logging

logger = logging.getLogger(__name__)


class Onnxruntime:
    pass


try:
    import onnxruntime  # noqa F401
    from onnxruntime.quantization import (
        QuantType,
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
    )
    from onnxruntime.transformers import optimizer
except ImportError:
    logger.warn(
        "onnxruntime module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    onnxruntime = Onnxruntime
    setattr(onnxruntime, "SessionOptions", None)
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = object
    optimizer = object
except FileNotFoundError:
    # Solves a colab issue
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = object
