from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.operations.conversions.converters import PytorchConverter
from nebullvm.operations.inference_learners.tvm import (
    PytorchApacheTVMInferenceLearner,
)
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.tvm import (
    ONNXApacheTVMCompiler,
    PyTorchApacheTVMCompiler,
)
from nebullvm.operations.optimizations.compilers.utils import tvm_is_available
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

device = (
    Device(DeviceType.GPU) if gpu_is_available() else Device(DeviceType.CPU)
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
            True,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        # (
        #     DeepLearningFramework.PYTORCH,
        #     True,
        #     QuantizationType.STATIC,
        #     2,
        #     "numeric_precision",
        # ),
    ],
)
@pytest.mark.skipif(
    not tvm_is_available(), reason="Apache TVM is not installed"
)
def test_tvm_onnx(
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

        compiler_op = ONNXApacheTVMCompiler()
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
            ModelCompiler.APACHE_TVM_ONNX
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
        assert isinstance(optimized_model, PytorchApacheTVMInferenceLearner)

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(loaded_model, PytorchApacheTVMInferenceLearner)

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

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = optimized_model(*inputs_example)
            assert res is not None

            res_orig = tuple(model(*inputs_example))
            assert all(
                [
                    torch.allclose(
                        res_tensor.float(), res_orig_tensor, rtol=1e-01
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
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
        ),
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.HALF,
            2,
            "numeric_precision",
        ),
        # (
        #     DeepLearningFramework.PYTORCH,
        #     True,
        #     QuantizationType.STATIC,
        #     2,
        #     "numeric_precision",
        # ),
    ],
)
@pytest.mark.skipif(
    not tvm_is_available(), reason="Can't test tvm if it's not installed."
)
def test_tvm_torch(
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
        ) = initialize_model(dynamic, metric, output_library, device)
        compiler_op = PyTorchApacheTVMCompiler()
        compiler_op.to(device).execute(
            model=model,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.APACHE_TVM_TORCH
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
        assert isinstance(optimized_model, PytorchApacheTVMInferenceLearner)

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = PytorchApacheTVMInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, PytorchApacheTVMInferenceLearner)

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
            res = optimized_model(*inputs_example)
            assert res is not None

            res_orig = tuple(model(*inputs_example))
            assert all(
                [
                    torch.allclose(
                        res_tensor.float(), res_orig_tensor, rtol=1e-01
                    )
                    for (res_tensor, res_orig_tensor) in zip(res, res_orig)
                ]
            )
