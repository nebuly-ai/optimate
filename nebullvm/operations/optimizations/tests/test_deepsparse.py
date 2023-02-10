from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.operations.inference_learners.deepsparse import (
    DEEPSPARSE_INFERENCE_LEARNERS,
)
from nebullvm.operations.measures.measures import MetricDropMeasure
from nebullvm.operations.measures.utils import compute_relative_difference
from nebullvm.operations.optimizations.base import (
    COMPILER_TO_INFERENCE_LEARNER_MAP,
)
from nebullvm.operations.optimizations.compilers.deepsparse import (
    DeepSparseCompiler,
)
from nebullvm.operations.optimizations.compilers.utils import (
    deepsparse_is_available,
)
from nebullvm.operations.optimizations.tests.utils import initialize_model
from nebullvm.operations.inference_learners.utils import load_model
from nebullvm.tools.base import (
    DeepLearningFramework,
    DeviceType,
    ModelCompiler,
    Device,
)

device = Device(DeviceType.CPU)


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        # (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
@pytest.mark.skipif(
    not deepsparse_is_available(),
    reason="Can't test deepsparse if it's not installed.",
)
def test_deepsparse(
    output_library: DeepLearningFramework,
    dynamic: bool,
    quantization_type=None,
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

        compiler_op = DeepSparseCompiler()
        compiler_op.to(device).execute(
            model=model,
            onnx_output_path=tmp_dir,
            model_params=model_params,
            quantization_type=None,
            input_data=input_data,
        )

        compiled_model = compiler_op.get_result()

        build_inference_learner_op = COMPILER_TO_INFERENCE_LEARNER_MAP[
            ModelCompiler.DEEPSPARSE
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
            optimized_model, DEEPSPARSE_INFERENCE_LEARNERS[output_library]
        )
        assert isinstance(optimized_model.get_size(), int)

        # Test save and load functions
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(
            loaded_model, DEEPSPARSE_INFERENCE_LEARNERS[output_library]
        )

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
        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )

        validity_check_op = MetricDropMeasure()
        validity_check_op.execute(
            optimized_model,
            test_input_data,
            model_outputs,
            CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )

        # Check validity of the optimized model
        assert validity_check_op.get_result()

        # Dynamic batch size is currently not supported from deepsparse
        # if dynamic:
        #     inputs_example = [
        #         input_[: len(input_) // 2] for input_ in inputs_example
        #     ]
        #     res = model(*inputs_example)
        #     assert res is not None
