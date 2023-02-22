from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.operations.inference_learners.neural_compressor import (
    NEURAL_COMPRESSOR_INFERENCE_LEARNERS,
)
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.intel_neural_compressor import (  # noqa: E501
    IntelNeuralCompressorCompiler,
)
from nebullvm.operations.optimizations.compilers.utils import (
    intel_neural_compressor_is_available,
)
from nebullvm.operations.optimizations.tests.utils import (
    initialize_model,
    check_model_validity,
)
from nebullvm.operations.inference_learners.utils import load_model
from nebullvm.tools.base import (
    DeepLearningFramework,
    QuantizationType,
    Device,
    ModelCompiler,
    DeviceType,
)

device = Device(DeviceType.CPU)


@pytest.mark.parametrize(
    ("output_library", "dynamic", "metric_drop_ths", "quantization_type"),
    [
        (DeepLearningFramework.PYTORCH, True, 2, QuantizationType.DYNAMIC),
        (DeepLearningFramework.PYTORCH, False, 2, QuantizationType.DYNAMIC),
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
        ) = initialize_model(dynamic, None, output_library, device)

        compiler_op = IntelNeuralCompressorCompiler()
        compiler_op.to(device).execute(
            model=model,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.INTEL_NEURAL_COMPRESSOR
        ]()

        build_inference_learner_op.to(device).execute(
            model=compiled_model,
            model_orig=compiler_op.model_orig
            if hasattr(compiler_op, "model_orig")
            else None,
            model_params=model_params,
            input_tfms=input_tfms,
            source_dl_framework=output_library,
        )

        optimized_model = build_inference_learner_op.get_result()

        assert isinstance(
            optimized_model,
            NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library],
        )

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(
            loaded_model, NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library]
        )

        assert isinstance(optimized_model.get_size(), int)

        inputs_example = optimized_model.get_inputs_example()
        res = optimized_model(*inputs_example)
        assert res is not None

        res_loaded = loaded_model(*inputs_example)
        assert all(
            [
                torch.allclose(res_tensor, res_loaded_tensor)
                for (res_tensor, res_loaded_tensor) in zip(res, res_loaded)
            ]
        )

        # Test validity of the model
        valid = check_model_validity(
            optimized_model,
            input_data,
            model_outputs,
            metric_drop_ths,
            quantization_type,
            metric,
        )
        assert valid

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None

            res_orig = tuple(model(*inputs_example))
            assert all(
                [
                    torch.allclose(res_tensor, res_orig_tensor, rtol=1e-01)
                    for (res_tensor, res_orig_tensor) in zip(res, res_orig)
                ]
            )
