from pathlib import Path
from typing import Union

from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.quantizations.intel_neural_compressor import (  # noqa: E501
    IntelNeuralCompressorQuantizer,
)
from nebullvm.operations.optimizations.quantizations.utils import (
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

        self.quantization_op = IntelNeuralCompressorQuantizer()

    def execute(
        self,
        model: Module,
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
        train_input_data = input_data.get_split("train")

        self.model_orig = model

        if quantization_type is not None:
            self.quantization_op.to(self.device).execute(
                model, quantization_type, input_tfms, train_input_data
            )
            model_quantized = self.quantization_op.get_result()
            self.compiled_model = self.compile_model(model_quantized)

    def compile_model(self, model: Union[str, Path]):
        return model
