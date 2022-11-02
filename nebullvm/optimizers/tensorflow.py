import logging
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Any

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.config import QUANTIZATION_DATA_NUM, CONSTRAINED_METRIC_DROP_THS
from nebullvm.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TF_BACKEND_LEARNERS_DICT,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.tensorflow import quantize_tf
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager

logger = logging.getLogger("nebullvm_logger")


class TensorflowBackendOptimizer(BaseOptimizer):
    """Optimizer working directly on the tensorflow backend, with no need of a
    conversion to ONNX. The model will be finally compiled using tflite.
    For avoiding un-wanted modification to the input model models are copied
    before being optimized.

    """

    def optimize(
        self,
        model: tf.Module,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        device: str,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[TensorflowBackendInferenceLearner]:
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (tf.Module): The tensorflow model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            output_library (DeepLearningFramework): Output framework. At the
                current stage just TENSORFLOW is supported.
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
            TensorflowBackendInferenceLearner or TFLiteBackendInferenceLearner:
                Model optimized for inference.
        """
        logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        assert output_library is DeepLearningFramework.TENSORFLOW, (
            "Other APIs than the Tensorflow one are not supported "
            "for the Tensorflow Backend yet."
        )

        check_quantization(quantization_type, metric_drop_ths)
        with TemporaryDirectory() as tmp_dir:
            train_input_data = input_data.get_split("train").get_list(
                QUANTIZATION_DATA_NUM
            )

            if quantization_type is not None:
                model, input_tfms = quantize_tf(
                    model=model,
                    quantization_type=quantization_type,
                    input_tfms=input_tfms,
                    input_data=train_input_data,
                    tmp_dir=tmp_dir,
                )

            learner = TF_BACKEND_LEARNERS_DICT[
                "tflite" if metric_drop_ths is not None else "tf"
            ](
                model,
                network_parameters=model_params,
                input_tfms=input_tfms,
                input_data=list(input_data.get_list(1)[0])
                if input_data is not None
                else None,
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
                        "The model optimized with Tensorflow backend gives a "
                        "different result compared with the original model. "
                        "This compiler will be skipped."
                    )
                return None
        return learner
