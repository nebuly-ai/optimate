from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, Any, List

import numpy as np
import tensorflow as tf
import torch

from nebullvm.base import ModelParams
from nebullvm.config import LEARNER_METADATA_FILENAME


@dataclass
class BaseInferenceLearner(ABC):
    network_parameters: ModelParams

    def predict_from_file(self, input_file: str, output_file: str):
        inputs = self._read_file(input_file)
        pred = self.predict(**inputs)
        self._save_file(pred, output_file)

    def predict_from_tensor(self, listified_tensor: List):
        inputs = self.list2tensor(listified_tensor)
        pred = self.predict(**inputs)
        return self.tensor2list(pred)

    def list2tensor(self, listified_tensor: List) -> Dict:
        raise NotImplementedError()

    def tensor2list(self, tensor: Any) -> List:
        raise NotImplementedError()

    def _read_file(self, input_file: str) -> Dict:
        raise NotImplementedError()

    def _save_file(self, prediction: Any, output_file: str):
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        """Take as input a tensor and returns a prediction"""
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_inputs_example(self):
        """The function returns a dictionary containing an example of the
        input for the optimized model predict method.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_format(self):
        return ".txt"

    @property
    @abstractmethod
    def input_format(self):
        return ".txt"


class LearnerMetadata:
    NAME: str = LEARNER_METADATA_FILENAME
    class_name: str
    module_name: str

    def __init__(
        self,
        class_name: str,
        module_name: str,
        network_parameters: Union[ModelParams, Dict],
        **kwargs,
    ):
        self.class_name = class_name
        self.module_name = module_name
        self.network_parameters = (
            network_parameters.dict()
            if isinstance(network_parameters, ModelParams)
            else network_parameters
        )
        self.__dict__.update(**kwargs)

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError(
                f"Error in key type. Expected str got {type(item)}"
            )
        elif item.startswith("_"):
            raise ValueError("Trying to access a private attribute.")

    @classmethod
    def from_model(cls, model: BaseInferenceLearner, **kwargs):
        return cls(
            class_name=model.__class__.__name__,
            module_name=model.__module__,
            network_parameters=model.network_parameters,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        if any(
            key not in dictionary
            for key in ("class_name", "module_name", "network_parameters")
        ):
            raise ValueError(
                "The input dictionary should contain both the model class "
                "name and module."
            )
        return cls(**dictionary)

    def to_dict(self) -> Dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if len(key) > 0 and key[0].islower() and not key.startswith("_")
        }

    @classmethod
    def read(cls, path: Union[Path, str]):
        path = Path(path)
        with open(path / cls.NAME, "r") as fin:
            metadata_dict = json.load(fin)
        return cls(**metadata_dict)

    def save(self, path: Union[Path, str]):
        path = Path(path)
        metadata_dict = self.to_dict()
        with open(path / self.NAME, "w") as fout:
            json.dump(metadata_dict, fout)

    def load_model(
        self, path: Union[Path, str], **kwargs
    ) -> BaseInferenceLearner:
        exec(f"from {self.module_name} import {self.class_name}")
        model = eval(self.class_name).load(path=path, **kwargs)
        return model


class PytorchBaseInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def input_format(self):
        return ".pt"

    @property
    def output_format(self):
        return ".pt"

    def list2tensor(self, listified_tensor: List) -> Dict:
        return {"input_tensor": torch.tensor(listified_tensor)}

    def tensor2list(self, tensor: torch.Tensor) -> List:
        return tensor.cpu().detach().numpy().tolist()

    def _read_file(self, input_file: Union[str, Path]) -> Dict:
        input_tensor = torch.load(input_file)
        return {"input_tensor": input_tensor}

    def _save_file(
        self, prediction: torch.Tensor, output_file: Union[str, Path]
    ):
        torch.save(prediction, output_file)

    def get_inputs_example(self):
        input_size = (
            self.network_parameters.batch_size,
            *self.network_parameters.input_size,
        )
        input_tensor = torch.randn(input_size)
        return {"input_tensor": input_tensor}


class TensorflowBaseInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def input_format(self):
        return ".npy"

    @property
    def output_format(self):
        return ".npy"

    def list2tensor(self, listified_tensor: List) -> Dict:
        return {"input_tensor": tf.convert_to_tensor(listified_tensor)}

    def tensor2list(self, tensor: tf.Tensor) -> List:
        return tensor.numpy().tolist()

    def _read_file(self, input_file: Union[str, Path]) -> Dict:
        numpy_array = np.load(input_file)
        input_tensor = tf.convert_to_tensor(numpy_array)
        return {"input_tensor": input_tensor}

    def _save_file(self, prediction: tf.Tensor, output_file: Union[str, Path]):
        prediction.numpy().save(output_file)

    def get_inputs_example(self):
        input_size = (
            self.network_parameters.batch_size,
            *self.network_parameters.input_size,
        )
        input_tensor = tf.random_normal_initializer()(shape=input_size)
        return {"input_tensor": input_tensor}
