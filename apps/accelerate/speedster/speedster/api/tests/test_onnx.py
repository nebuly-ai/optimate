import cpuinfo
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.operations.inference_learners.onnx import (
    NumpyONNXInferenceLearner,
)
from nebullvm.operations.inference_learners.openvino import (
    NumpyOpenVinoInferenceLearner,
)
from nebullvm.operations.inference_learners.tensor_rt import (
    NumpyONNXTensorRTInferenceLearner,
)
from nebullvm.operations.inference_learners.tvm import (
    NumpyApacheTVMInferenceLearner,
)
from nebullvm.operations.optimizations.compilers.utils import tvm_is_available
from torchvision import models

from speedster import optimize_model, load_model
from speedster.api.tests.utils import torch_to_onnx


def test_onnx_ort():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18()
        input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]
        model_path = torch_to_onnx(model, input_data, tmp_dir)

        input_data = [
            ((np.random.randn(1, 3, 256, 256).astype(np.float32),), 0)
            for i in range(100)
        ]

        # Run nebullvm optimization in one line of code
        optimized_model = optimize_model(
            model_path,
            input_data=input_data,
            ignore_compilers=[
                compiler
                for compiler in COMPILER_LIST
                if compiler != "onnxruntime"
            ],
            ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
            # metric_drop_ths=2,
        )

        with TemporaryDirectory() as tmp_dir:
            optimized_model.save(tmp_dir)
            loaded_model = load_model(tmp_dir)
            assert isinstance(loaded_model, NumpyONNXInferenceLearner)

            assert isinstance(loaded_model.get_size(), int)

            # Try the optimized model
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            x = torch.randn(1, 3, 256, 256, requires_grad=False)
            model.eval()
            res_original = model(x.to(device))
            res_optimized = optimized_model(x.numpy())[0]

            assert (
                abs(
                    (res_original.detach().cpu().numpy() - res_optimized)
                ).max()
                < 1e-2
            )


def test_onnx_ort_quant():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18()
        input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]
        model_path = torch_to_onnx(model, input_data, tmp_dir)

        input_data = [
            ((np.random.randn(1, 3, 256, 256).astype(np.float32),), 0)
            for i in range(100)
        ]

        # Run nebullvm optimization in one line of code
        optimized_model = optimize_model(
            model_path,
            input_data=input_data,
            ignore_compilers=[
                compiler
                for compiler in COMPILER_LIST
                if compiler != "onnxruntime"
            ],
            ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
            metric_drop_ths=2,
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyONNXInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Skip because cuda is not available.",
)
def test_onnx_tensorrt():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18()
        input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]
        model_path = torch_to_onnx(model, input_data, tmp_dir)

        input_data = [
            ((np.random.randn(1, 3, 256, 256).astype(np.float32),), 0)
            for i in range(100)
        ]

        # Run nebullvm optimization in one line of code
        optimized_model = optimize_model(
            model_path,
            input_data=input_data,
            ignore_compilers=[
                compiler
                for compiler in COMPILER_LIST
                if compiler != "tensor_rt"
            ],
            ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        model.eval()
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyONNXTensorRTInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-2
        )


@pytest.mark.skipif(
    "intel" not in cpuinfo.get_cpu_info()["brand_raw"].lower(),
    reason="Openvino is only available for intel processors.",
)
def test_onnx_openvino():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18()
        input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]
        model_path = torch_to_onnx(model, input_data, tmp_dir)

        input_data = [
            ((np.random.randn(1, 3, 256, 256).astype(np.float32),), 0)
            for i in range(100)
        ]

        # Run nebullvm optimization in one line of code
        optimized_model = optimize_model(
            model_path,
            input_data=input_data,
            ignore_compilers=[
                compiler
                for compiler in COMPILER_LIST
                if compiler != "openvino"
            ],
            ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
            device="cpu",
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        model.eval()
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyOpenVinoInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-2
        )


@pytest.mark.skipif(
    not tvm_is_available(), reason="Can't test tvm if it's not installed."
)
def test_onnx_tvm():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18()
        input_data = [((torch.randn(1, 3, 256, 256),), 0) for i in range(100)]
        model_path = torch_to_onnx(model, input_data, tmp_dir)

        input_data = [
            ((np.random.randn(1, 3, 256, 256).astype(np.float32),), 0)
            for i in range(100)
        ]

        # Run nebullvm optimization in one line of code
        optimized_model = optimize_model(
            model_path,
            input_data=input_data,
            ignore_compilers=[
                compiler for compiler in COMPILER_LIST if compiler != "tvm"
            ],
            ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        model.eval()
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyApacheTVMInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-2
        )
