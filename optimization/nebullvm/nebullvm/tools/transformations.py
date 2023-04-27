import copy
from abc import ABC, abstractmethod
from typing import List, Any, Dict

import numpy as np

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch


class BaseTransformation(ABC):
    @abstractmethod
    def _transform(self, _input: Any, **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, _input: Any, **kwargs):
        return self._transform(_input, **kwargs)

    def to_dict(self):
        return {
            "module": self.__class__.__module__,
            "name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, tfm_dict: Dict):
        return cls()


class MultiStageTransformation(BaseTransformation):
    def __init__(self, transformations: List[BaseTransformation]):
        self._tfms = transformations

    def _transform(self, _input: Any, **kwargs) -> Any:
        for tfm in self._tfms:
            _input = tfm(_input, **kwargs)
        return _input

    def append(self, __tfm: BaseTransformation):
        self._tfms.append(__tfm)

    def extend(self, tfms: List[BaseTransformation]):
        self._tfms += tfms

    def to_dict(self) -> Dict:
        return {"tfms": [tfm.to_dict() for tfm in self._tfms]}

    def to_list(self):
        return self._tfms

    @classmethod
    def from_dict(cls, tfms_dict: Dict):
        tfms = []
        for tfm_dict in tfms_dict["tfms"]:
            exec(f"from {tfm_dict['module']} import {tfm_dict['name']}")
            tfm = eval(tfm_dict["name"]).from_dict(tfm_dict)
            tfms.append(tfm)
        return cls(tfms)

    def copy(self):
        new_list = copy.deepcopy(self._tfms)
        return self.__class__(new_list)

    def __len__(self):
        return len(self._tfms)


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


class NoOp(BaseTransformation):
    def _transform(self, _input: Any, **kwargs):
        return _input


class VerifyContiguity(BaseTransformation):
    def _transform(self, _input: Any, **kwargs) -> Any:
        if not isinstance(_input, torch.Tensor):
            return _input
        if not _input.is_contiguous():
            _input = _input.contiguous()
        return _input
