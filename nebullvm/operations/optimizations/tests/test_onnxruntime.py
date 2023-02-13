import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import pytest
import torch

from nebullvm.operations.conversions.converters import PytorchConverter
from nebullvm.operations.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.onnxruntime import (
    ONNXCompiler,
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
        "external_data_format",
    ),
    [
        (DeepLearningFramework.PYTORCH, True, None, None, None, True),
        (DeepLearningFramework.PYTORCH, True, None, None, None, False),
        (DeepLearningFramework.PYTORCH, False, None, None, None, False),
    ],
)
def test_onnxruntime(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
    external_data_format: bool,
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

        # Test onnx external data format (large models)
        if external_data_format:
            onnx_model = onnx.load(model_path)
            onnx.save_model(
                onnx_model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
            )

        compiler_op = ONNXCompiler()
        compiler_op.to(device).execute(
            model=model_path,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.ONNX_RUNTIME
        ]()
        build_inference_learner_op.to(device).execute(
            model=compiled_model,
            model_orig=compiler_op.model_orig
            if hasattr(compiler_op, "model_orig")
            else None,
            model_params=model_params,
            input_tfms=input_tfms,
            source_dl_framework=output_library,
            quantization_type=quantization_type,
        )

        optimized_model = build_inference_learner_op.get_result()
        assert isinstance(
            optimized_model, ONNX_INFERENCE_LEARNERS[output_library]
        )

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(
            loaded_model, ONNX_INFERENCE_LEARNERS[output_library]
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
                    torch.allclose(res_tensor, res_orig_tensor, rtol=2e-01)
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
        "external_data_format",
    ),
    [
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.DYNAMIC,
            2,
            "numeric_precision",
            False,
        ),
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.STATIC,
            2,
            "numeric_precision",
            False,
        ),
    ],
)
@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="onnxruntime with int8 precision is very slow on GPU",
)
def test_onnxruntime_quantization(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
    external_data_format: bool,
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

        # Test onnx external data format (large models)
        if external_data_format:
            onnx_model = onnx.load(model_path)
            onnx.save_model(
                onnx_model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
            )

        compiler_op = ONNXCompiler()
        compiler_op.to(device).execute(
            model=model_path,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.ONNX_RUNTIME
        ]()
        build_inference_learner_op.to(device).execute(
            model=compiled_model,
            model_orig=compiler_op.model_orig
            if hasattr(compiler_op, "model_orig")
            else None,
            model_params=model_params,
            input_tfms=input_tfms,
            source_dl_framework=output_library,
            quantization_type=quantization_type,
        )

        optimized_model = build_inference_learner_op.get_result()
        assert isinstance(
            optimized_model, ONNX_INFERENCE_LEARNERS[output_library]
        )

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(
            loaded_model, ONNX_INFERENCE_LEARNERS[output_library]
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
                    torch.allclose(res_tensor, res_orig_tensor, rtol=2e-01)
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
        "external_data_format",
    ),
    [
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.HALF,
            2,
            "numeric_precision",
            False,
        ),
        (
            DeepLearningFramework.PYTORCH,
            True,
            QuantizationType.HALF,
            2,
            "numeric_precision",
            True,
        ),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="onnxruntime with half precision on windows does not work",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="onnxruntime with half precision is very slow on CPU",
)
def test_onnxruntime_half(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type: QuantizationType,
    metric_drop_ths: int,
    metric: str,
    external_data_format: bool,
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

        # Test onnx external data format (large models)
        if external_data_format:
            onnx_model = onnx.load(model_path)
            onnx.save_model(
                onnx_model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=False,
            )

        compiler_op = ONNXCompiler()
        compiler_op.to(device).execute(
            model=model_path,
            input_tfms=input_tfms,
            metric_drop_ths=metric_drop_ths,
            quantization_type=quantization_type,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.ONNX_RUNTIME
        ]()
        build_inference_learner_op.to(device).execute(
            model=compiled_model,
            model_orig=compiler_op.model_orig
            if hasattr(compiler_op, "model_orig")
            else None,
            model_params=model_params,
            input_tfms=input_tfms,
            source_dl_framework=output_library,
            quantization_type=quantization_type,
        )

        optimized_model = build_inference_learner_op.get_result()
        assert isinstance(
            optimized_model, ONNX_INFERENCE_LEARNERS[output_library]
        )

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = ONNX_INFERENCE_LEARNERS[output_library].load(tmp_dir)
        assert isinstance(
            loaded_model, ONNX_INFERENCE_LEARNERS[output_library]
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
                        res_tensor.float(), res_orig_tensor, rtol=1e-01
                    )
                    for (res_tensor, res_orig_tensor) in zip(res, res_orig)
                ]
            )
