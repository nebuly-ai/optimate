from pathlib import Path
from tempfile import TemporaryDirectory

import cpuinfo
import pytest
import torch

from nebullvm.operations.conversions.converters import PytorchConverter
from nebullvm.operations.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
)
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.openvino import (
    OpenVINOCompiler,
)
from nebullvm.operations.optimizations.tests.utils import (
    initialize_model,
    check_model_validity,
)
from nebullvm.operations.inference_learners.utils import load_model
from nebullvm.tools.base import (
    Device,
    DeepLearningFramework,
    QuantizationType,
    ModelCompiler,
    DeviceType,
)


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
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
        ),
    ],
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

        model_path = Path(tmp_dir) / "fp32"
        model_path.mkdir(parents=True)

        converter_op = PytorchConverter()
        converter_op.to(device).set_state(model, input_data).execute(
            model_path, model_params
        )

        converted_models = converter_op.get_result()
        assert len(converted_models) > 1

        model_path = str(
            [model for model in converted_models if isinstance(model, Path)][0]
        )

        compiler_op = OpenVINOCompiler()
        compiler_op.to(device).execute(
            model=model_path,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.OPENVINO
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
            optimized_model, OPENVINO_INFERENCE_LEARNERS[output_library]
        )

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(
            loaded_model, OPENVINO_INFERENCE_LEARNERS[output_library]
        )

        assert isinstance(optimized_model.get_size(), int)

        inputs_example = list(optimized_model.get_inputs_example())
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

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = optimized_model(*inputs_example)
            assert res is not None

            res_orig = tuple(model(*inputs_example))
            assert all(
                [
                    torch.allclose(
                        res_tensor.float(), res_orig_tensor, rtol=2e-01
                    )
                    for (res_tensor, res_orig_tensor) in zip(res, res_orig)
                ]
            )
