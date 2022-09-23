import os
from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.converters.torch_converters import convert_torch_to_onnx
from nebullvm.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
)
from nebullvm.inference_learners.tensor_rt import NVIDIA_INFERENCE_LEARNERS
from nebullvm.optimizers import TensorRTOptimizer
from nebullvm.optimizers.tests.utils import initialize_model


@pytest.mark.parametrize(
    (
        "output_library",
        "dynamic",
        "quantization_type",
        "metric_drop_ths",
        "metric",
    ),
    [
        (DeepLearningFramework.PYTORCH, True, None, None, None),
        (DeepLearningFramework.PYTORCH, False, None, None, None),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
        ),
    ],
)
def test_tensorrt_onnx(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return
    elif quantization_type == QuantizationType.DYNAMIC:
        # Dynamic quantization is not supported
        return None
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric_drop_ths, metric, output_library)

        model_path = os.path.join(tmp_dir, "test_model.onnx")
        convert_torch_to_onnx(model, model_params, model_path)
        optimizer = TensorRTOptimizer()
        model = optimizer.optimize(
            model=model_path,
            output_library=output_library,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            metric=metric,
            input_data=input_data,
            model_outputs=model_outputs,
        )
        assert isinstance(model, NVIDIA_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = NVIDIA_INFERENCE_LEARNERS[output_library].load(tmp_dir)
        assert isinstance(
            loaded_model, NVIDIA_INFERENCE_LEARNERS[output_library]
        )

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None


@pytest.mark.parametrize(
    (
        "output_library",
        "dynamic",
        "quantization_type",
        "metric_drop_ths",
        "metric",
    ),
    [
        (DeepLearningFramework.PYTORCH, True, None, None, None),
        (DeepLearningFramework.PYTORCH, False, None, None, None),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
        ),
    ],
)
def test_tensorrt_torch(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return
    elif quantization_type == QuantizationType.DYNAMIC:
        # Dynamic quantization is not supported
        return None
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric_drop_ths, metric, output_library)
        optimizer = TensorRTOptimizer()
        model = optimizer.optimize_from_torch(
            torch_model=model,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            metric=metric,
            input_data=input_data,
            model_outputs=model_outputs,
        )
        assert isinstance(model, PytorchTensorRTInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = PytorchTensorRTInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, PytorchTensorRTInferenceLearner)

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None
