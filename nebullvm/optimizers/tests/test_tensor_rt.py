import pytest
import torch

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.tensor_rt import NVIDIA_INFERENCE_LEARNERS
from nebullvm.optimizers import TensorRTOptimizer
from nebullvm.optimizers.tests.utils import get_onnx_model


@pytest.mark.parametrize("output_library", [DeepLearningFramework.PYTORCH])
def test_tensor_rt(output_library: DeepLearningFramework):
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return
    model_path, model_params = get_onnx_model()
    optimizer = TensorRTOptimizer()
    model = optimizer.optimize(model_path, output_library, model_params)
    assert isinstance(model, NVIDIA_INFERENCE_LEARNERS[output_library])

    res = model.predict(*model.get_inputs_example())
    assert res is not None
