from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.optimizers.onnx import ONNXOptimizer
from nebullvm.optimizers.tests.utils import get_onnx_model


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
def test_onnxruntime(output_library: DeepLearningFramework, dynamic: bool):
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_onnx_model(tmp_dir, dynamic)
        optimizer = ONNXOptimizer()
        model = optimizer.optimize(model_path, output_library, model_params)
        assert isinstance(model, ONNX_INFERENCE_LEARNERS[output_library])

        inputs_example = list(model.get_inputs_example())
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
