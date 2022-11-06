from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.deepsparse import (
    DEEPSPARSE_INFERENCE_LEARNERS,
)
from nebullvm.optimizers.deepsparse import DeepSparseOptimizer
from nebullvm.optimizers.tests.utils import initialize_model
from nebullvm.utils.compilers import deepsparse_is_available
from nebullvm.utils.general import gpu_is_available


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
def test_deepsparse(output_library: DeepLearningFramework, dynamic: bool):
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            input_data,
            model_params,
            input_tfms,
            model_outputs,
            metric,
        ) = initialize_model(dynamic, None, output_library)

        device = "gpu" if gpu_is_available() else "cpu"
        optimizer = DeepSparseOptimizer()
        model, metric_drop = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            input_data=input_data,
            model_outputs=model_outputs,
            device=device,
        )
        assert isinstance(model, DEEPSPARSE_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = DEEPSPARSE_INFERENCE_LEARNERS[output_library].load(
            tmp_dir
        )
        assert isinstance(
            loaded_model, DEEPSPARSE_INFERENCE_LEARNERS[output_library]
        )

        inputs_example = model.get_inputs_example()
        res = model(*inputs_example)
        assert res is not None

        # Dynamic batch size is currently not supported from deepsparse
        # if dynamic:
        #     inputs_example = [
        #         input_[: len(input_) // 2] for input_ in inputs_example
        #     ]
        #     res = model(*inputs_example)
        #     assert res is not None
