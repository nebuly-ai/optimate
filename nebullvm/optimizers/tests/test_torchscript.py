from tempfile import TemporaryDirectory

import torch

import pytest

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.installers.installers import get_cpu_arch, _get_os
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
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
def test_torchscript(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    if (
        _get_os() == "Darwin"
        and get_cpu_arch() == "arm"
        and quantization_type is not None
    ):
        # Quantization doesn't work on M1 chips
        return
    elif (
        not torch.cuda.is_available()
        and quantization_type == QuantizationType.HALF
    ):
        # Half quantization fails on CPU
        return
    elif torch.cuda.is_available() and quantization_type in [
        QuantizationType.STATIC,
        QuantizationType.DYNAMIC,
    ]:
        # Quantization is not supported on GPU
        return

    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric_drop_ths, metric, output_library)

        optimizer = PytorchBackendOptimizer()
        model = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            metric=metric,
            input_data=input_data,
            model_outputs=model_outputs,
        )
        assert isinstance(model, PytorchBackendInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = PytorchBackendInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, PytorchBackendInferenceLearner)

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None
