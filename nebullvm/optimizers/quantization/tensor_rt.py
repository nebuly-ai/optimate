from typing import List, Tuple

import numpy as np
import torch

from nebullvm.config import NO_COMPILER_INSTALLATION

if torch.cuda.is_available():
    try:
        import polygraphy
        from tensorrt import IInt8EntropyCalibrator2
    except ImportError:
        import warnings

        if not NO_COMPILER_INSTALLATION:
            from nebullvm.installers.installers import install_tensor_rt

            warnings.warn(
                "No TensorRT valid installation has been found. "
                "Trying to install it from source."
            )
            install_tensor_rt()
            from tensorrt import IInt8EntropyCalibrator2
        else:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "It won't be possible to use it in the following steps."
            )
            IInt8EntropyCalibrator2 = object
else:
    IInt8EntropyCalibrator2 = object


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
            cuda_data = [
                polygraphy.cuda.DeviceArray().copy_from(
                    host_buffer=input_tensor, stream=cuda_stream
                )
                for input_tensor in data
            ]
            return [input_tensor.ptr for input_tensor in cuda_data]
        except StopIteration:
            return None

    def get_batch_size(self):
        return self._bs

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None
