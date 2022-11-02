import logging
from typing import Optional, Callable, Any

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.config import QUANTIZATION_DATA_NUM, CONSTRAINED_METRIC_DROP_THS
from nebullvm.inference_learners.onnx import (
    ONNXInferenceLearner,
    ONNX_INFERENCE_LEARNERS,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.onnx import quantize_onnx
from nebullvm.optimizers.quantization.utils import (
    check_precision,
    check_quantization,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.logger import (
    save_root_logger_state,
    load_root_logger_state,
    raise_logger_level,
    debug_mode_enabled,
)
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
)

logger = logging.getLogger("nebullvm_logger")


class ONNXOptimizer(BaseOptimizer):
    """Class for compiling the AI models using ONNX runtime."""

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        device: str,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[ONNXInferenceLearner]:
        """Build the ONNX runtime learner from the onnx model.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            device: (str): Device where the model will be run.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction. Default: None.
            input_data (DataManager, optional): User defined data.
                Default: None.
            model_outputs (Any, optional): Outputs computed by the original
                model. Default: None.

        Returns:
            ONNXInferenceLearner: Model running on onnxruntime. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        check_quantization(quantization_type, metric_drop_ths)

        input_data_onnx = input_data.get_split("train").get_numpy_list(
            QUANTIZATION_DATA_NUM
        )

        if not debug_mode_enabled():
            logger_state = save_root_logger_state()
            raise_logger_level()

        use_gpu = device == "gpu"

        if quantization_type is not None:
            model, input_tfms = quantize_onnx(
                model, quantization_type, input_tfms, input_data_onnx, use_gpu
            )

        if not debug_mode_enabled():
            load_root_logger_state(logger_state)

        learner = ONNX_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            onnx_path=model,
            input_names=get_input_names(model),
            output_names=get_output_names(model),
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
            device=device,
        )

        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )
        is_valid = check_precision(
            learner,
            test_input_data,
            model_outputs,
            metric_drop_ths
            if quantization_type is not None
            else CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )

        if not is_valid:
            if quantization_type is None:
                logger.warning(
                    "The model optimized with onnxruntime gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped."
                )
            return None
        return learner
