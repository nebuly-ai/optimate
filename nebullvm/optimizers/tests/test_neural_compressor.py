from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.inference_learners.neural_compressor import (
    NEURAL_COMPRESSOR_INFERENCE_LEARNERS,
)
from nebullvm.optimizers.neural_compressor import NeuralCompressorOptimizer
from nebullvm.optimizers.tests.utils import initialize_model
from nebullvm.utils.compilers import intel_neural_compressor_is_available


@pytest.mark.parametrize(
    ("output_library", "dynamic", "metric_drop_ths", "quantization_type"),
    [
        (DeepLearningFramework.PYTORCH, True, 2, QuantizationType.DYNAMIC),
        (DeepLearningFramework.PYTORCH, False, 2, QuantizationType.DYNAMIC),
        (DeepLearningFramework.PYTORCH, True, 2, QuantizationType.HALF),
        (DeepLearningFramework.PYTORCH, False, 2, QuantizationType.HALF),
        (DeepLearningFramework.PYTORCH, True, 2, QuantizationType.STATIC),
        (DeepLearningFramework.PYTORCH, False, 2, QuantizationType.STATIC),
    ],
)
@pytest.mark.skipif(
    not intel_neural_compressor_is_available(),
    reason="Can't test neural compressor if it's not installed.",
)
def test_neural_compressor(
    output_library: DeepLearningFramework,
    dynamic: bool,
    metric_drop_ths: float,
    quantization_type: QuantizationType,
):
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, None, output_library)

        optimizer = NeuralCompressorOptimizer()
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
        assert isinstance(
            model, NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library]
        )

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = NEURAL_COMPRESSOR_INFERENCE_LEARNERS[
            output_library
        ].load(tmp_dir)
        assert isinstance(
            loaded_model, NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library]
        )

        inputs_example = model.get_inputs_example()
        res = model(*inputs_example)
        assert res is not None

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None
