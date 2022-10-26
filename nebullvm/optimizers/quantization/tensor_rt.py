from typing import List, Tuple

import numpy as np

from nebullvm.optional_modules.tensor_rt import (
    IInt8EntropyCalibrator2,
    polygraphy,
)


class TensorRTCalibrator(IInt8EntropyCalibrator2):
    def __init__(
        self, batch_size: int, input_data: List[Tuple[np.ndarray, ...]]
    ):
        super(TensorRTCalibrator, self).__init__()
        self._bs = batch_size
        self.batches = (x for x in input_data)

    def get_batch(self, names):
        cuda_stream = polygraphy.cuda.Stream()
        try:
            data = next(self.batches)

            cuda_data = []
            for input_tensor in data:
                device_array = polygraphy.cuda.DeviceArray()
                device_array.resize(input_tensor.shape)
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
