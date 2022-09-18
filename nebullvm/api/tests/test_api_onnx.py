import cpuinfo
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torchvision import models

from nebullvm.api.functions import optimize_model
from nebullvm.api.tests.utils import torch_to_onnx
from nebullvm.inference_learners.onnx import NumpyONNXInferenceLearner
from nebullvm.inference_learners.tensor_rt import (
    NumpyNvidiaInferenceLearner,
)
from nebullvm.inference_learners.openvino import (
    NumpyOpenVinoInferenceLearner,
)
from nebullvm.inference_learners.tvm import NumpyApacheTVMInferenceLearner
from nebullvm.utils.compilers import (
    tvm_is_available,
)


def test_onnx_onnx():
    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18().eval()
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
                "deepsparse",
                "tvm",
                "torchscript",
                "tensor RT",
                "openvino",
                "bladedisc",
            ],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyONNXInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-5
        )


def test_onnx_tensorrt():
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return

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
                "deepsparse",
                "tvm",
                "torchscript",
                "onnxruntime",
                "openvino",
                "bladedisc",
            ],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        model.eval()
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyNvidiaInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-5
        )


def test_onnx_openvino():
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        return

    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18().eval()
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
                "deepsparse",
                "tvm",
                "torchscript",
                "onnxruntime",
                "tensor RT",
                "bladedisc",
            ],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyOpenVinoInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-5
        )


def test_onnx_tvm():
    if not tvm_is_available():
        return None

    with TemporaryDirectory() as tmp_dir:
        model = models.resnet18().eval()
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
                "deepsparse",
                "tensor RT",
                "torchscript",
                "onnxruntime",
                "openvino",
                "bladedisc",
            ],
        )

        # Try the optimized model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 256, 256, requires_grad=False)
        res_original = model(x.to(device))
        res_optimized = optimized_model(x.numpy())[0]

        assert isinstance(optimized_model, NumpyApacheTVMInferenceLearner)
        assert (
            abs((res_original.detach().cpu().numpy() - res_optimized)).max()
            < 1e-5
        )
