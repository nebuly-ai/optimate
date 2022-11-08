from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.optimizers.tensorflow import TensorflowBackendOptimizer
from nebullvm.optimizers.tests.utils import initialize_model
from nebullvm.utils.general import gpu_is_available


@pytest.mark.parametrize(
    (
        "output_library",
        "dynamic",
        "quantization_type",
        "metric_drop_ths",
        "metric",
    ),
    [
        (DeepLearningFramework.TENSORFLOW, True, None, None, None),
        (DeepLearningFramework.TENSORFLOW, False, None, None, None),
        (
            DeepLearningFramework.TENSORFLOW,
            False,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            False,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            False,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
        ),
    ],
)
def test_tensorflow(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):

    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric, output_library)

        device = "gpu" if gpu_is_available() else "cpu"
        optimizer = TensorflowBackendOptimizer()
        model, metric_drop = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            metric=metric,
            input_data=input_data,
            model_outputs=model_outputs,
            device=device,
        )

        if quantization_type is None:
            assert isinstance(model, TensorflowBackendInferenceLearner)
        else:
            assert isinstance(model, TFLiteBackendInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        if quantization_type is None:
            loaded_model = TensorflowBackendInferenceLearner.load(tmp_dir)
            assert isinstance(loaded_model, TensorflowBackendInferenceLearner)
        else:
            loaded_model = TFLiteBackendInferenceLearner.load(tmp_dir)
            assert isinstance(loaded_model, TFLiteBackendInferenceLearner)

        assert isinstance(model.get_size(), int)

        inputs_example = list(model.get_inputs_example())
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
