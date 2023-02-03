from typing import List, Tuple

import numpy as np

from nebullvm.optional_modules.tensor_rt import (
    tensorrt as trt,
    IInt8EntropyCalibrator2,
    polygraphy,
)
from nebullvm.tools.base import QuantizationType, ModelParams
from nebullvm.tools.transformations import (
    MultiStageTransformation,
)


def quantize_tensorrt(
    quantization_type: QuantizationType,
    model_params: ModelParams,
    config,
    input_tfms: MultiStageTransformation,
    input_data: List[Tuple[np.ndarray, ...]] = None,
):
    if quantization_type is QuantizationType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
        # Tensor RT does not need to transform input data
        # to fp16 because it expects always fp32
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

    return config


class TensorRTCalibrator(IInt8EntropyCalibrator2):
    def __init__(
        self, batch_size: int, input_data: List[Tuple[np.ndarray, ...]]
    ):
        super(TensorRTCalibrator, self).__init__()
        self._bs = batch_size
        self.batches = (x for x in input_data)

    def get_batch(self, names):
        cuda_stream = polygraphy.Stream()
        try:
            data = next(self.batches)

            cuda_data = []
            for input_tensor in data:
                device_array = polygraphy.DeviceArray(
                    shape=input_tensor.shape, dtype=input_tensor.dtype
                )
                device_array.copy_from(
                    host_buffer=input_tensor, stream=cuda_stream
                )
                cuda_data.append(device_array)

            return [input_tensor.ptr for input_tensor in cuda_data]
        except StopIteration:
            return None

    def get_batch_size(self):
        return self._bs

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None
