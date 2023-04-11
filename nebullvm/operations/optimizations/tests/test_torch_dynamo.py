import platform
from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.core.models import (
    DeviceType,
    Device,
    DeepLearningFramework,
    QuantizationType,
    ModelCompiler,
)
from nebullvm.operations.inference_learners.torch_dynamo import (
    TorchDynamoInferenceLearner,
)
from nebullvm.operations.optimizations.compilers.torch_dynamo import (
    TorchDynamoCompiler,
)
from nebullvm.operations.optimizations.optimizers.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.tests.utils import (
    initialize_model,
    check_model_validity,
)
from nebullvm.tools.utils import gpu_is_available, check_module_version

device = (
    Device(DeviceType.GPU) if gpu_is_available() else Device(DeviceType.CPU)
)


def run_test_torch_dynamo(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    with TemporaryDirectory() as tmp_dir:  # noqa: F841
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, metric, output_library, device)

        compiler_op = TorchDynamoCompiler()
        compiler_op.to(device).execute(
            model=model,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
            model_params=model_params,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.TORCH_DYNAMO
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
        assert isinstance(optimized_model, TorchDynamoInferenceLearner)

        # Test save and load functions
        # optimized_model.save(tmp_dir)
        # loaded_model = load_model(tmp_dir)
        # assert isinstance(loaded_model, TorchDynamoInferenceLearner)

        assert isinstance(optimized_model.get_size(), int)

        inputs_example = list(optimized_model.get_inputs_example())
        res = optimized_model(*inputs_example)
        assert res is not None

        # res_loaded = loaded_model(*inputs_example)
        # assert all(
        #     [
        #         torch.allclose(res_tensor, res_loaded_tensor)
        #         for (res_tensor, res_loaded_tensor) in zip(res, res_loaded)
        #     ]
        # )

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
            torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            inputs_example = [
                input_[: len(input_) // 2].to(torch_device)
                for input_ in inputs_example
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
    ],
)
@pytest.mark.skipif(
    not check_module_version(torch, min_version="2.0.0"),
    reason="Torch version is not supported",
)
@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Torch compile() is not currently supported on windows",
)
def test_torch_dynamo_fp32(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
):
    run_test_torch_dynamo(
        output_library,
        dynamic,
        quantization_type,
        metric_drop_ths,
        metric,
    )
