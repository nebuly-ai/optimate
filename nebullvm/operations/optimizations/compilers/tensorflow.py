from typing import List, Tuple

from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.compilers.base import Compiler

from nebullvm.operations.optimizations.compilers.quantizations.tensorflow import (  # noqa: E501
    quantize_tensorflow,
)
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class TensorflowBackendCompiler(Compiler):
    supported_ops = {
        "cpu": [None],
        "gpu": [None],
    }

    def execute(
        self,
        model: tf.Module,
        input_data: DataManager,
        input_tfms: MultiStageTransformation,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            input_data (DataManager): User defined data. Default: None.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.

        Returns:
            PytorchBackendInferenceLearner: Model optimized for inference.
        """

        if quantization_type not in self.supported_ops[self.device.value]:
            self.compiled_model = None
            return

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        check_quantization(quantization_type, metric_drop_ths)

        self.compiled_model = model

    def compile_model(self):
        pass

    @staticmethod
    def quantize_model(**kwargs):
        raise NotImplementedError()


class TFLiteBackendCompiler(Compiler):
    supported_ops = {
        "cpu": [
            None,
            QuantizationType.STATIC,
            QuantizationType.HALF,
            QuantizationType.DYNAMIC,
        ],
        "gpu": [],
    }

    def execute(
        self,
        model: tf.Module,
        input_data: DataManager,
        input_tfms: MultiStageTransformation,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            input_data (DataManager): User defined data. Default: None.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.

        Returns:
            PytorchBackendInferenceLearner: Model optimized for inference.
        """

        if quantization_type not in self.supported_ops[self.device.value]:
            self.compiled_model = None
            return

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        check_quantization(quantization_type, metric_drop_ths)
        train_input_data = input_data.get_split("train").get_list(
            QUANTIZATION_DATA_NUM
        )

        if quantization_type is not None:
            self.compiled_model = self.quantize_model(
                model, quantization_type, input_tfms, train_input_data
            )
        else:
            self.compiled_model = self.compile_model(model)

    def compile_model(
        self,
        model: tf.Module,
    ):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def quantize_model(
        model: tf.Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data_tensorflow: List[Tuple[tf.Tensor, ...]],
    ):
        return quantize_tensorflow(
            model, quantization_type, input_tfms, input_data_tensorflow
        )
