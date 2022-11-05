try:
    import onnx  # noqa F401
except ImportError:
    onnx = None

try:
    import onnxmltools  # noqa F401
    from onnxmltools.utils.float16_converter import (  # noqa F401
        convert_float_to_float16_model_path,
    )

except ImportError:
    convert_float_to_float16_model_path = object
