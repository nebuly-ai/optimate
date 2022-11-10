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
except ImportError:
    tensorflow = Tensorflow


try:
    import tf2onnx  # noqa F401
except ImportError:
    tf2onnx = object
