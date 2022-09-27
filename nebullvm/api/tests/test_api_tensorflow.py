import cpuinfo

import tensorflow as tf
import torch
from tensorflow.keras.applications.resnet50 import ResNet50

from nebullvm.api.functions import optimize_model
from nebullvm.inference_learners.onnx import TensorflowONNXInferenceLearner
from nebullvm.inference_learners.tensor_rt import (
    TensorflowNvidiaInferenceLearner,
)
from nebullvm.inference_learners.openvino import (
    TensorflowOpenVinoInferenceLearner,
)
from nebullvm.inference_learners.tvm import TensorflowApacheTVMInferenceLearner
from nebullvm.utils.compilers import (
    tvm_is_available,
)

# Limit tensorflow gpu memory usage
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def test_tensorflow_onnx():
    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            "deepsparse",
            "tvm",
            "torchscript",
            "tensor RT",
            "openvino",
            "bladedisc",
        ],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    tf.keras.clear_session()

    assert isinstance(optimized_model, TensorflowONNXInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


def test_tensorflow_tensorrt():
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return

    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            "deepsparse",
            "tvm",
            "torchscript",
            "onnxruntime",
            "openvino",
            "bladedisc",
        ],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    tf.keras.clear_session()

    assert isinstance(optimized_model, TensorflowNvidiaInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


def test_tensorflow_openvino():
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        return

    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            "deepsparse",
            "tvm",
            "torchscript",
            "onnxruntime",
            "tensor RT",
            "bladedisc",
        ],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    tf.keras.clear_session()

    assert isinstance(optimized_model, TensorflowOpenVinoInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


def test_tensorflow_tvm():
    if not tvm_is_available():
        return None

    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            "deepsparse",
            "openvino",
            "torchscript",
            "onnxruntime",
            "tensor RT",
            "bladedisc",
        ],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    tf.keras.clear_session()

    assert isinstance(optimized_model, TensorflowApacheTVMInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2
