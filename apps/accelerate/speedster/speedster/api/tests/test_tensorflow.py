from tempfile import TemporaryDirectory

import cpuinfo
import pytest
import tensorflow as tf
from keras.applications import ResNet50
from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.operations.inference_learners.onnx import (
    TensorflowONNXInferenceLearner,
)
from nebullvm.operations.inference_learners.openvino import (
    TensorflowOpenVinoInferenceLearner,
)
from nebullvm.operations.inference_learners.tensor_rt import (
    TensorflowONNXTensorRTInferenceLearner,
)
from nebullvm.operations.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.tvm import (
    TensorflowApacheTVMInferenceLearner,
)
from nebullvm.operations.optimizations.compilers.utils import tvm_is_available
from nebullvm.tools.utils import gpu_is_available

from speedster import optimize_model, load_model

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

    with TemporaryDirectory() as tmp_dir:
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(loaded_model, TensorflowONNXInferenceLearner)

        assert isinstance(loaded_model.get_size(), int)

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowONNXInferenceLearner)
    assert abs((res_original - res_optimized)).max() < 1e-2


def test_tensorflow_tf_backend():
    model = ResNet50()
    input_data = [
        ((tf.random.normal([1, 224, 224, 3]),), 0) for i in range(100)
    ]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "xla"
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
    gpu_is_available(),
    reason="TFLite does not support Nvidia GPUs",
)
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
        metric_drop_ths=0.1,
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TFLiteBackendInferenceLearner)
    assert abs((res_original - res_optimized)).max() < 1e-2


@pytest.mark.skipif(
    not gpu_is_available(),
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
            compiler for compiler in COMPILER_LIST if compiler != "tensor_rt"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowONNXTensorRTInferenceLearner)
    assert abs((res_original - res_optimized)).max() < 1e-2


@pytest.mark.skipif(
    "intel" not in cpuinfo.get_cpu_info()["brand_raw"].lower(),
    reason="Openvino is only available for intel processors.",
)
def test_tensorflow_openvino():
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
        device="cpu",
    )

    # Try the optimized model
    x = tf.random.normal([1, 224, 224, 3])
    res_original = model.predict(x)
    res_optimized = optimized_model.predict(x)[0]

    assert isinstance(optimized_model, TensorflowOpenVinoInferenceLearner)
    assert abs((res_original - res_optimized)).max() < 1e-2


@pytest.mark.skipif(
    not tvm_is_available(), reason="Can't test tvm if it's not installed."
)
def test_tensorflow_tvm():
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
    assert abs((res_original - res_optimized)).max() < 1e-2
