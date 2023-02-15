from tempfile import TemporaryDirectory

import pytest

from nebullvm.operations.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.tensorflow import (
    TensorflowBackendCompiler,
    TFLiteBackendCompiler,
)
from nebullvm.operations.optimizations.tests.utils import (
    initialize_model,
    check_model_validity,
)
from nebullvm.operations.inference_learners.utils import load_model
from nebullvm.tools.base import (
    DeepLearningFramework,
    QuantizationType,
    DeviceType,
    ModelCompiler,
    Device,
)
from nebullvm.tools.utils import gpu_is_available


@pytest.mark.parametrize(
    (
        "output_library",
        "dynamic",
        "quantization_type",
        "metric_drop_ths",
        "metric",
    ),
    [
        (DeepLearningFramework.TENSORFLOW, False, None, None, None),
        (DeepLearningFramework.TENSORFLOW, True, None, None, None),
    ],
)
def test_tensorflow_backend(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    device = (
        Device(DeviceType.GPU)
        if gpu_is_available()
        else Device(DeviceType.CPU)
    )
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric, output_library, device)

        compiler_op = TensorflowBackendCompiler()
        compiler_op.to(device).execute(
            model=model,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.XLA
        ]()

        build_inference_learner_op.to(device).execute(
            model=compiled_model,
            model_orig=compiler_op.model_orig
            if hasattr(compiler_op, "model_orig")
            else None,
            model_params=model_params,
            input_tfms=input_tfms,
            dl_framework=output_library,
        )

        optimized_model = build_inference_learner_op.get_result()

        assert isinstance(optimized_model, TensorflowBackendInferenceLearner)

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(loaded_model, TensorflowBackendInferenceLearner)

        assert isinstance(optimized_model.get_size(), int)

        inputs_example = list(optimized_model.get_inputs_example())
        res = optimized_model.predict(*inputs_example)
        assert res is not None

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

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = optimized_model.predict(*inputs_example)
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
        (
            DeepLearningFramework.TENSORFLOW,
            False,
            None,
            0.1,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            True,
            None,
            0.1,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            True,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            True,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.TENSORFLOW,
            True,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
        ),
    ],
)
def test_tf_lite(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    device = Device(DeviceType.CPU)
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric, output_library, device)

        compiler_op = TFLiteBackendCompiler()
        compiler_op.to(device).execute(
            model=model,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.TFLITE
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

        assert isinstance(optimized_model, TFLiteBackendInferenceLearner)

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = TFLiteBackendInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, TFLiteBackendInferenceLearner)

        assert isinstance(optimized_model.get_size(), int)

        inputs_example = list(optimized_model.get_inputs_example())
        res = optimized_model.predict(*inputs_example)
        assert res is not None

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

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = optimized_model.predict(*inputs_example)
            assert res is not None
