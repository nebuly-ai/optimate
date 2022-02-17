import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
)
from nebullvm.optimizers.openvino import OpenVinoOptimizer
from nebullvm.optimizers.tests.utils import get_onnx_model


@pytest.mark.parametrize("output_library", [DeepLearningFramework.PYTORCH])
def test_openvino(output_library: DeepLearningFramework):
    model_path, model_params = get_onnx_model()
    optimizer = OpenVinoOptimizer()
    model = optimizer.optimize(model_path, output_library, model_params)
    assert isinstance(model, OPENVINO_INFERENCE_LEARNERS[output_library])

    res = model.predict(*model.get_inputs_example())
    assert res is not None
