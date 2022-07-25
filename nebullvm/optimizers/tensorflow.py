from tempfile import TemporaryDirectory
from typing import Callable, Optional

import tensorflow as tf

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TF_BACKEND_LEARNERS_DICT,
)
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.tensorflow import quantize_tf
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import convert_to_target_framework
from nebullvm.utils.tf import create_model_inputs_tf, run_tf_model


class TensorflowBackendOptimizer(BaseOptimizer):
    """Optimizer working directly on the tensorflow backend, with no need of a
    conversion to ONNX. The model will be finally compiled using tflite.
    For avoiding un-wanted modification to the input model models are copied
    before being optimized.

    Attributes:
        logger (Logger, optional): Optional logger for logging optimization
            information.
    """

    def optimize(
        self,
        model: tf.Module,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> Optional[TensorflowBackendInferenceLearner]:
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (tf.Module): The tensorflow model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            output_library (DeepLearningFramework): Output framework. At the
                current stage just TENSORFLOW is supported.
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
            TensorflowBackendInferenceLearner or TFLiteBackendInferenceLearner:
                Model optimized for inference.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        assert output_library is DeepLearningFramework.TENSORFLOW, (
            "Other APIs than the Tensorflow one are not supported "
            "for the Tensorflow Backend yet."
        )

        check_quantization(quantization_type, metric_drop_ths)
        with TemporaryDirectory() as tmp_dir:
            if metric_drop_ths is not None:
                if input_data is None:
                    input_data_tf = [
                        tuple(
                            create_model_inputs_tf(
                                model_params.batch_size,
                                model_params.input_infos,
                            )
                        )
                    ]
                    ys = None
                else:
                    input_data_tf, ys = input_data.get_numpy_list(
                        300, with_ys=True
                    )
                    input_data_tf = [
                        tuple(
                            convert_to_target_framework(t, output_library)
                            for t in data_tuple
                        )
                        for data_tuple in input_data_tf
                    ]
                output_data_tf = [
                    tuple(run_tf_model(model, input_tensors))
                    for input_tensors in input_data_tf
                ]
                model, input_tfms = quantize_tf(
                    model=model,
                    quantization_type=quantization_type,
                    input_tfms=input_tfms,
                    input_data=input_data_tf,
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
            if metric_drop_ths is not None:
                is_valid = check_precision(
                    learner,
                    input_data_tf,
                    output_data_tf,
                    metric_drop_ths,
                    metric_func=metric,
                    ys=ys,
                )
                if not is_valid:
                    return None
        return learner
