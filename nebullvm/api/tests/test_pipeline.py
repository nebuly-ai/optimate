import cpuinfo

import torch
import torchvision.models as models

from nebullvm.api.functions import optimize_model
from nebullvm.inference_learners.onnx import PytorchONNXInferenceLearner
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
    PytorchNvidiaInferenceLearner,
)
from nebullvm.inference_learners.openvino import (
    PytorchOpenVinoInferenceLearner,
)
from nebullvm.inference_learners.tvm import PytorchApacheTVMInferenceLearner
from nebullvm.utils.compilers import (
    tvm_is_available,
    bladedisc_is_available,
)


def test_torch_onnx():
    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchONNXInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5


def test_torch_torchscript():
    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            "deepsparse",
            "tvm",
            "onnxruntime",
            "tensor RT",
            "openvino",
            "bladedisc",
        ],
    )

    # Try the optimized model
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchBackendInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5


def test_torch_tensorrt():
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return

    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(
        optimized_model, PytorchTensorRTInferenceLearner
    ) or isinstance(optimized_model, PytorchNvidiaInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5


def test_torch_openvino():
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        return

    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchOpenVinoInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5


def test_torch_tvm():
    if not tvm_is_available():
        return None

    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchApacheTVMInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5


def test_torch_bladedisc():
    if not bladedisc_is_available():
        return None

    model = models.resnet18().eval()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
            "tvm",
        ],
    )

    # Try the optimized model
    x = torch.randn(1, 3, 256, 256)
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchApacheTVMInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-5