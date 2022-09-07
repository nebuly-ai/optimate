import warnings

from nebullvm.installers.installers import install_tf2onnx, install_tensorflow
from nebullvm.utils.general import check_module_version


class Keras:
    Model = None


class Tensorflow:
    Module = None
    Tensor = None
    keras = Keras()

    @staticmethod
    def function(**kwargs):
        return lambda x: x


try:
    import tensorflow  # noqa F401

    if not check_module_version(tensorflow, min_version="2.7.0"):
        warnings.warn(
            "tensorflow module version must be >= 2.7.0. "
            "Trying to update it."
        )
        install_tensorflow()
        import tensorflow
except ImportError:
    warnings.warn(
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
            "tf2onnx module is not installed on this platform. "
            "Trying to install it."
        )

        install_tf2onnx()
        import tf2onnx
    except ImportError:
        tf2onnx = object
