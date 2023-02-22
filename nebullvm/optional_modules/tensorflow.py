try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass


class Model:
    pass


class Dataset:
    pass


class Keras:
    Model = Model


class Module:
    pass


class Tensor:
    pass


class data:
    Dataset = Dataset


class Tensorflow:
    Module = Module
    Tensor = Tensor
    keras = Keras()
    data = data

    @staticmethod
    def function(**kwargs):
        return lambda x: x


try:
    import tensorflow  # noqa F401

    physical_devices = tensorflow.config.experimental.list_physical_devices(
        "GPU"
    )
    if len(physical_devices) > 0:
        for physical_device in physical_devices:
            tensorflow.config.experimental.set_memory_growth(
                physical_device, True
            )

    tensorflow.get_logger().setLevel("ERROR")
    tensorflow.autograph.set_verbosity(0)
except (ImportError, AttributeError):
    tensorflow = Tensorflow


try:
    import tf2onnx  # noqa F401

    tf2onnx.logging.set_level("ERROR")
    tf2onnx.logging.set_tf_verbosity("ERROR")
except ImportError:
    tf2onnx = object
