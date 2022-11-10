from typing import Any

import numpy as np

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.transformations.base import BaseTransformation


class HalfPrecisionTransformation(BaseTransformation):
    @staticmethod
    def _transform_numpy(_input: np.ndarray) -> np.ndarray:
        return _input.astype(dtype=np.float16)

    @staticmethod
    def _transform_tf(_input: tf.Tensor) -> tf.Tensor:
        return tf.cast(_input, tf.float16)

    @staticmethod
    def _transform_torch(_input: torch.Tensor) -> torch.Tensor:
        return _input.half()

    def _transform(self, _input: Any, **kwargs) -> Any:
        if isinstance(_input, np.ndarray):
            return (
                self._transform_numpy(_input)
                if _input.dtype == np.float32
                else _input
            )
        elif isinstance(_input, torch.Tensor):
            return (
                self._transform_torch(_input)
                if _input.dtype == torch.float32
                else _input
            )
        elif isinstance(_input, tf.Tensor) and _input is not None:
            return (
                self._transform_tf(_input)
                if _input.dtype == tf.float32
                else _input
            )
        else:
            raise TypeError(
                f"The given input type is not currently supported. "
                f"Got {type(_input)}, expected one between (np.ndarray, "
                f"torch.Tensor, tf.Tensor)"
            )
