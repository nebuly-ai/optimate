import warnings

from nebullvm.installers.installers import install_tf2onnx
from nebullvm.utils.general import check_module_version

NoneType = type(None)


class Keras:
    Model = NoneType


class Tensorflow:
    Module = NoneType
    Tensor = NoneType
    keras = Keras()

    @staticmethod
    def function(**kwargs):
        return lambda x: x


try:
    import tensorflow  # noqa F401

    if not check_module_version(tensorflow, min_version="2.7.0"):
        warnings.warn(
            "tensorflow module version must be >= 2.7.0. "
            "Please update it if you want to use it."
        )
except ImportError:
    warnings.warn(
        "Missing Library: "
        "tensorflow module is not installed on this platform."
        "Please install it if you want to use tensorflow API."
    )
    tensorflow = Tensorflow


try:
    import tf2onnx  # noqa F401
except ImportError:
    try:
        import tensorflow  # noqa F401

        warnings.warn(
            "Missing Library: "
            "tf2onnx module is not installed on this platform. "
            "Please install it if you want to exploit the ONNX "
            "pipeline with a Tensorflow model."
        )

        install_tf2onnx()
        import tf2onnx
    except ImportError:
        tf2onnx = object


try:
    import onnxruntime as ort  # noqa F401
except ImportError:
    warnings.warn(
        "Missing Library: "
        "onnxruntime module is not installed on this platform."
        "Please install it if you want to use it."
    )
