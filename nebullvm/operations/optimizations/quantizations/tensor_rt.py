from typing import Any, List, Tuple

import numpy as np

from nebullvm.base import ModelParams, QuantizationType
from nebullvm.operations.optimizations.quantizations.base import Quantizer
from nebullvm.optimizers.quantization.tensor_rt import TensorRTCalibrator
from nebullvm.optional_modules.tensor_rt import tensorrt as trt


class ONNXTensorRTQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.config = None

    def execute(
        self,
        quantization_type: QuantizationType,
        model_params: ModelParams,
        config,
        input_data: List[Tuple[np.ndarray, ...]] = None,
    ):
        if quantization_type is QuantizationType.HALF:
            config.set_flag(trt.BuilderFlag.FP16)
        elif quantization_type is QuantizationType.STATIC:
            assert input_data is not None, (
                "You need to specify the calibration data for "
                "performing static quantization."
            )
            calibrator = TensorRTCalibrator(
                batch_size=model_params.batch_size,
                input_data=input_data,
            )
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator

        self.config = config

    def get_result(self) -> Any:
        return self.config
