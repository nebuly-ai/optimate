from typing import Optional

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.inference_learners.onnx import (
    ONNXInferenceLearner,
    ONNX_INFERENCE_LEARNERS,
)
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.onnx import quantize_onnx
from nebullvm.optimizers.quantization.utils import (
    check_precision,
    check_quantization,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
    create_model_inputs_onnx,
    run_onnx_model,
    convert_to_target_framework,
)


class ONNXOptimizer(BaseOptimizer):
    """Class for creating the inference learner running on onnxruntime."""

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        quantization_ths: float = None,
        quantization_type: QuantizationType = None,
    ) -> Optional[ONNXInferenceLearner]:
        """Build the ONNX runtime learner from the onnx model.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            quantization_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.

        Returns:
            ONNXInferenceLearner: Model running on onnxruntime. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        input_data_onnx, output_data_onnx = [], []
        check_quantization(quantization_type, quantization_ths)
        if quantization_ths is not None:
            input_data_onnx = [
                tuple(
                    create_model_inputs_onnx(
                        model_params.batch_size, model_params.input_infos
                    )
                )
            ]
            output_data_onnx = [
                tuple(run_onnx_model(onnx_model, list(input_tensors)))
                for input_tensors in input_data_onnx
            ]
            onnx_model, input_tfms = quantize_onnx(
                onnx_model, quantization_type, input_tfms, input_data_onnx
            )
        learner = ONNX_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            onnx_path=onnx_model,
            input_names=get_input_names(onnx_model),
            output_names=get_output_names(onnx_model),
        )
        if quantization_ths is not None:
            input_data = [
                tuple(
                    convert_to_target_framework(t, output_library)
                    for t in data_tuple
                )
                for data_tuple in input_data_onnx
            ]
            is_valid = check_precision(
                learner, input_data, output_data_onnx, quantization_ths
            )
            if not is_valid:
                return None
        return learner
