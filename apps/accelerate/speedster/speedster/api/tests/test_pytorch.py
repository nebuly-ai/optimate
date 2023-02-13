import cpuinfo
from tempfile import TemporaryDirectory

import pytest
import torch
import torchvision.models as models
from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.operations.inference_learners.blade_disc import (
    BladeDISCInferenceLearner,
)
from nebullvm.operations.inference_learners.onnx import (
    PytorchONNXInferenceLearner,
)
from nebullvm.operations.inference_learners.openvino import (
    PytorchOpenVinoInferenceLearner,
)
from nebullvm.operations.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
    PytorchONNXTensorRTInferenceLearner,
)
from nebullvm.operations.inference_learners.tvm import (
    PytorchApacheTVMInferenceLearner,
)
from nebullvm.operations.optimizations.compilers.utils import (
    tvm_is_available,
    bladedisc_is_available,
)

from speedster import optimize_model, load_model


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

    with TemporaryDirectory() as tmp_dir:
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(loaded_model, PytorchONNXInferenceLearner)

        assert isinstance(loaded_model.get_size(), int)

    # Try the optimized model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256, requires_grad=False).to(device)
    model.to(device).eval()
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
    model.to(device).eval()
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
    model.to(device).eval()
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
            compiler for compiler in COMPILER_LIST if compiler != "tensor_rt"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
    )

    # Try the optimized model
    x = torch.randn(1, 3, 256, 256).cuda()
    model.cuda().eval()
    res_original = model(x)
    res_optimized = optimized_model(x)[0]

    assert isinstance(
        optimized_model, PytorchTensorRTInferenceLearner
    ) or isinstance(optimized_model, PytorchONNXTensorRTInferenceLearner)
    assert torch.max(abs((res_original - res_optimized))) < 1e-2


@pytest.mark.skipif(
    "intel" not in cpuinfo.get_cpu_info()["brand_raw"].lower(),
    reason="Openvino is only available for intel processors.",
)
def test_torch_openvino():
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
    model.to(device).eval()
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
