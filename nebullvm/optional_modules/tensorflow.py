import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

NoneType = type(None)


class Keras:
    Model = NoneType


class data:
    Dataset = NoneType


class Tensorflow:
    Module = NoneType
    Tensor = NoneType
    keras = Keras()
    data = data

    @staticmethod
    def function(**kwargs):
        return lambda x: x


try:
    import tensorflow  # noqa F401

    tensorflow.get_logger().setLevel("ERROR")
    tensorflow.autograph.set_verbosity(0)
except ImportError:
    tensorflow = Tensorflow


try:
    import tf2onnx  # noqa F401

    tf2onnx.logging.set_level("ERROR")
    tf2onnx.logging.set_tf_verbosity("ERROR")
except ImportError:
    tf2onnx = object
