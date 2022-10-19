import cpuinfo

import pytest
import tensorflow as tf
import torch
from tensorflow.keras.applications.resnet50 import ResNet50

from nebullvm.api.functions import optimize_model
from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.inference_learners.onnx import TensorflowONNXInferenceLearner
from nebullvm.inference_learners.openvino import (
    TensorflowOpenVinoInferenceLearner,
)
from nebullvm.inference_learners.tensor_rt import (
    TensorflowNvidiaInferenceLearner,
)
from nebullvm.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
)
from nebullvm.inference_learners.tvm import TensorflowApacheTVMInferenceLearner
from nebullvm.utils.compilers import (
    tvm_is_available,
)
from nebullvm.utils.general import is_python_version_3_10

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


def test_tensorflow_ort():
    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowONNXInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


def test_tensorflow_tflite():
    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "tflite"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowBackendInferenceLearner)
    assert abs((res_original - res_optimized)).max() < 1e-2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Skip because cuda is not available.",
)
def test_tensorflow_tensorrt():
    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "tensor RT"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowNvidiaInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


@pytest.mark.skipif(
    is_python_version_3_10(), reason="Openvino doesn't support python 3.10 yet"
)
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
            compiler for compiler in COMPILER_LIST if compiler != "openvino"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowOpenVinoInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2


@pytest.mark.skipif(
    not tvm_is_available(), reason="Can't test tvm if it's not installed."
)
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
            compiler for compiler in COMPILER_LIST if compiler != "tvm"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowApacheTVMInferenceLearner)
    assert abs((res_original - res_optimized)).numpy().max() < 1e-2
