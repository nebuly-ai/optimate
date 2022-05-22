import copy
from abc import ABC, abstractmethod
from typing import List, Any, Dict


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


class NoOp(BaseTransformation):
    def _transform(self, _input: Any, **kwargs):
        return _input
