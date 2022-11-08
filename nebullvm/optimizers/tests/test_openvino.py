from pathlib import Path
from tempfile import TemporaryDirectory

import cpuinfo
import pytest

from nebullvm.base import DeepLearningFramework, QuantizationType
from nebullvm.converters.torch_converters import convert_torch_to_onnx
from nebullvm.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
)
from nebullvm.optimizers.openvino import OpenVinoOptimizer
from nebullvm.optimizers.tests.utils import initialize_model
from nebullvm.utils.general import is_python_version_3_10, gpu_is_available


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
@pytest.mark.skipif(
    is_python_version_3_10(),
    reason="Openvino doesn't support python 3.10 yet.",
)
@pytest.mark.skipif(
    "intel" not in cpuinfo.get_cpu_info()["brand_raw"].lower(),
    reason="Openvino is only available for intel processors.",
)
def test_openvino(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    if quantization_type == QuantizationType.DYNAMIC:
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
        ) = initialize_model(dynamic, metric, output_library)

        model_path = Path(tmp_dir) / "fp32"
        model_path.mkdir(parents=True)
        model_path = str(model_path / "test_model.onnx")
        device = "gpu" if gpu_is_available() else "cpu"
        convert_torch_to_onnx(model, model_params, model_path, device)
        optimizer = OpenVinoOptimizer()
        model, metric_drop = optimizer.optimize(
            model=model_path,
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
        assert isinstance(model, OPENVINO_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = OPENVINO_INFERENCE_LEARNERS[output_library].load(
            tmp_dir
        )
        assert isinstance(
            loaded_model, OPENVINO_INFERENCE_LEARNERS[output_library]
        )

        assert isinstance(model.get_size(), int)

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model(*inputs_example)
            assert res is not None
