from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.inference_learners.onnx import (
    ONNXInferenceLearner,
    ONNX_INFERENCE_LEARNERS,
)
from nebullvm.optimizers import BaseOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.onnx import get_input_names, get_output_names


class ONNXOptimizer(BaseOptimizer):
    """Class for creating the inference learner running on onnxruntime."""

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
    ) -> ONNXInferenceLearner:
        """Build the ONNX runtime learner from the onnx model.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.

        Returns:
            ONNXInferenceLearner: Model running on onnxruntime. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        learner = ONNX_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            onnx_path=onnx_model,
            input_names=get_input_names(onnx_model),
            output_names=get_output_names(onnx_model),
        )
        return learner
