import cpuinfo

import pytest
import torch
import torchvision.models as models

from nebullvm import optimize_model
from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.inference_learners.blade_disc import BladeDISCInferenceLearner
from nebullvm.inference_learners.onnx import PytorchONNXInferenceLearner
from nebullvm.inference_learners.openvino import (
    PytorchOpenVinoInferenceLearner,
)
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
    PytorchNvidiaInferenceLearner,
)
from nebullvm.inference_learners.tvm import PytorchApacheTVMInferenceLearner
from nebullvm.utils.compilers import (
    tvm_is_available,
    bladedisc_is_available,
)
from nebullvm.utils.general import is_python_version_3_10


def test_torch_ort():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchONNXInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2


def test_torch_ort_quant():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        metric_drop_ths=2,
    )

    # Try the optimized model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchONNXInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 2


def test_torch_torchscript():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "torchscript"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchBackendInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Skip because cuda is not available.",
)
def test_torch_tensorrt():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256).cuda()
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(
        optimized_model, PytorchTensorRTInferenceLearner
    ) or isinstance(optimized_model, PytorchNvidiaInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2


@pytest.mark.skipif(
    is_python_version_3_10(), reason="Openvino doesn't support python 3.10 yet"
)
def test_torch_openvino():
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        return

    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchOpenVinoInferenceLearner)
    assert torch.max(abs((res_original.cpu() - res_optimized))) < 1e-2


@pytest.mark.skipif(
    not tvm_is_available(), reason="Can't test tvm if it's not installed."
)
def test_torch_tvm():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, PytorchApacheTVMInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2


@pytest.mark.skipif(
    not bladedisc_is_available(),
    reason="Can't test bladedisc if it's not installed.",
)
def test_torch_bladedisc():
    model = models.resnet18()
    input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]

    # Run nebullvm optimization in one line of code
    optimized_model = optimize_model(
        model,
        input_data=input_data,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "bladedisc"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(optimized_model, BladeDISCInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2
