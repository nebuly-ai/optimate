from pathlib import Path
from typing import Union

from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.quantizations.intel_neural_compressor import (  # noqa: E501
    quantize_neural_compressor,
)
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.torch import Module
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class IntelNeuralCompressorCompiler(Compiler):
    supported_ops = {
        "cpu": [
            QuantizationType.STATIC,
            QuantizationType.DYNAMIC,
        ],
        "gpu": [],
    }

    def __init__(self):
        super().__init__()
        self.model_orig = None

    def execute(
        self,
        model: Module,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Compile the input model using IntelNeuralCompressor library.

        Args:
            model (torch.nn.Module): The pytorch model.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with a higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            input_data (DataManager): User defined data. Default: None
        """

        if quantization_type not in self.supported_ops[self.device.type.value]:
            self.compiled_model = None
            return

        if quantization_type is QuantizationType.STATIC and input_data is None:
            raise ValueError("Input data is required for static quantization.")

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        check_quantization(quantization_type, metric_drop_ths)
        train_input_data = input_data.get_split("train")

        self.model_orig = model

        if quantization_type is not None:
            quantized_model = self._quantize_model(
                model, quantization_type, input_tfms, train_input_data
            )
            self.compiled_model = self._compile_model(quantized_model)

    def _compile_model(self, model: Union[str, Path]):
        return model

    @staticmethod
    def _quantize_model(
        model: Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data: DataManager,
    ):
        return quantize_neural_compressor(
            model, quantization_type, input_tfms, input_data
        )
