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
except ImportError:
    onnxruntime = Onnxruntime
    setattr(onnxruntime, "SessionOptions", None)
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = object
except FileNotFoundError:
    # Solves a colab issue
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = object

try:
    # They require torch
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.optimizer import MODEL_TYPES
except ImportError:
    MODEL_TYPES = object
    optimizer = object
