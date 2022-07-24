from typing import Optional, Callable

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
from nebullvm.utils.data import DataManager
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
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> Optional[ONNXInferenceLearner]:
        """Build the ONNX runtime learner from the onnx model.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            ONNXInferenceLearner: Model running on onnxruntime. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        input_data_onnx, output_data_onnx, ys = [], [], None
        check_quantization(quantization_type, metric_drop_ths)
        if metric_drop_ths is not None:
            if input_data is None:
                input_data_onnx = [
                    tuple(
                        create_model_inputs_onnx(
                            model_params.batch_size, model_params.input_infos
                        )
                    )
                ]
            else:
                input_data_onnx, ys = input_data.get_numpy_list(
                    300, with_ys=True
                )
            output_data_onnx = [
                tuple(run_onnx_model(model, list(input_tensors)))
                for input_tensors in input_data_onnx
            ]
            model, input_tfms = quantize_onnx(
                model, quantization_type, input_tfms, input_data_onnx
            )

        learner = ONNX_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            onnx_path=model,
            input_names=get_input_names(model),
            output_names=get_output_names(model),
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )
        if metric_drop_ths is not None:
            inputs = [
                tuple(
                    convert_to_target_framework(t, output_library)
                    for t in data_tuple
                )
                for data_tuple in input_data_onnx
            ]
            is_valid = check_precision(
                learner,
                inputs,
                output_data_onnx,
                metric_drop_ths,
                metric_func=metric,
                ys=ys,
            )
            if not is_valid:
                return None
        return learner
