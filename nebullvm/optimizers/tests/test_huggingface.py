from tempfile import TemporaryDirectory

import pytest

from nebullvm.api.huggingface import HuggingFaceInferenceLearner
from nebullvm.base import DeepLearningFramework
from nebullvm.optimizers.extra import HuggingFaceOptimizer
from nebullvm.optimizers.tests.utils import get_huggingface_model
from nebullvm.utils.general import gpu_is_available


@pytest.mark.parametrize(
    ("output_library",),
    [(DeepLearningFramework.PYTORCH,)],
)
def test_huggingface(output_library: DeepLearningFramework):
    with TemporaryDirectory() as tmp_dir:
        (
            model,
            model_params,
            output_structure,
            input_names,
            output_type,
            input_data,
            model_outputs,
        ) = get_huggingface_model(tmp_dir, output_library)

        device = "gpu" if gpu_is_available() else "cpu"
        optimizer = HuggingFaceOptimizer({})
        model, metric_drop = optimizer.optimize(
            model,
            output_library,
            model_params,
            input_data=input_data,
            model_outputs=model_outputs,
            device=device,
        )

        model = HuggingFaceInferenceLearner(
            core_inference_learner=model,
            output_structure=output_structure,
            input_names=input_names,
            output_type=output_type,
        )

        assert isinstance(model, HuggingFaceInferenceLearner)
        assert isinstance(model.get_size(), int)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = HuggingFaceInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, HuggingFaceInferenceLearner)

        inputs_example = list(model.get_inputs_example())
        res = model(*inputs_example)
        assert res is not None
