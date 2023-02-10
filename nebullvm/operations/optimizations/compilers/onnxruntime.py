from pathlib import Path
from typing import Union, List, Tuple

import numpy as np

from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.compilers.base import Compiler

from nebullvm.operations.optimizations.compilers.quantizations.onnx import (
    quantize_onnx,
)
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class ONNXCompiler(Compiler):
    supported_ops = {
        "cpu": [
            None,
            QuantizationType.STATIC,
            QuantizationType.DYNAMIC,
        ],
        "gpu": [
            None,
            QuantizationType.HALF,
        ],
    }

    def execute(
        self,
        model: str,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Compile the input model using ONNX Runtime Compiler.

        Args:
            model (str): The onnx model path.
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
        train_input_data = input_data.get_split("train").get_numpy_list(
            QUANTIZATION_DATA_NUM
        )

        if quantization_type is not None:
            model = self._quantize_model(
                model, train_input_data, quantization_type, input_tfms
            )

        self.compiled_model = self._compile_model(model)

    def _compile_model(self, model: Union[str, Path]):
        return model

    def _quantize_model(
        self,
        model_path: str,
        input_data: List[Tuple[np.ndarray, ...]],
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
    ):
        return quantize_onnx(
            model_path, input_data, quantization_type, self.device, input_tfms
        )
