from pathlib import Path
from tempfile import TemporaryDirectory

import onnx

import pytest

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.converters.torch_converters import convert_torch_to_onnx
from nebullvm.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.optimizers.onnx import ONNXOptimizer
from nebullvm.optimizers.tests.utils import initialize_model


@pytest.mark.parametrize(
    (
        "output_library",
        "dynamic",
        "quantization_type",
        "metric_drop_ths",
        "metric",
        "external_data",
    ),
    [
        (DeepLearningFramework.PYTORCH, True, None, None, None, True),
        (DeepLearningFramework.PYTORCH, True, None, None, None, False),
        (DeepLearningFramework.PYTORCH, False, None, None, None, False),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
            False,
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.HALF,
            2,
            "numeric_precision",
            False,
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.HALF,
            2,
            "numeric_precision",
            True,
        ),
        (
            DeepLearningFramework.PYTORCH,
            False,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
            False,
        ),
    ],
)
def test_onnxruntime(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
    external_data: bool,
):
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric_drop_ths, metric, output_library)

        model_path = Path(tmp_dir) / "fp32"
        model_path.mkdir(parents=True)
        model_path = str(model_path / "test_model.onnx")
        convert_torch_to_onnx(model, model_params, model_path)

        # Test onnx external data format (large models)
        if external_data:
            onnx_model = onnx.load(model_path)
            onnx.save_model(
                onnx_model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
            )

        optimizer = ONNXOptimizer()
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
        assert isinstance(model, ONNX_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = ONNX_INFERENCE_LEARNERS[output_library].load(tmp_dir)
        assert isinstance(
            loaded_model, ONNX_INFERENCE_LEARNERS[output_library]
        )

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None
