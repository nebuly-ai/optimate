from nebullvm.optional_modules.dummy import DummyClass

try:
    import onnxruntime  # noqa F401
    from onnxruntime.quantization import (
        QuantType,
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
    )
except ImportError:
    onnxruntime = DummyClass
    setattr(onnxruntime, "SessionOptions", None)
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = DummyClass
except FileNotFoundError:
    # Solves a colab issue
    QuantType = quantize_static = quantize_dynamic = None
    CalibrationDataReader = DummyClass

try:
    # They require torch
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.optimizer import MODEL_TYPES
except ImportError:
    MODEL_TYPES = DummyClass
    optimizer = DummyClass
