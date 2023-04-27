from nebullvm.optional_modules.dummy import DummyClass

try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass


class Keras:
    Model = DummyClass


class data:
    Dataset = DummyClass


class dtypes:
    DType = DummyClass


class Tensorflow:
    Module = DummyClass
    Tensor = DummyClass
    keras = Keras()
    data = data
    dtypes = dtypes
    float16 = float32 = int32 = int64 = DummyClass

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
