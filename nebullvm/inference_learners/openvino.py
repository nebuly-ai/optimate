import shutil
import warnings
from abc import ABC
import json
from pathlib import Path
from typing import Dict, Union, Type

import numpy as np
import tensorflow as tf
import torch

from nebullvm.config import OPENVINO_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
)
from nebullvm.base import ModelParams, DeepLearningFramework

try:
    from openvino.inference_engine import IECore
except ImportError:
    warnings.warn(
        "No Openvino library detected. "
        "The Openvino Inference learner should not be used."
    )


class OpenVinoInferenceLearner(BaseInferenceLearner, ABC):
    def __init__(
        self,
        exec_network,
        input_key,
        output_key,
        description_file: str,
        weights_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.exec_network = exec_network
        self.input_key = input_key
        self.output_key = output_key
        self.description_file = description_file
        self.weights_file = weights_file

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        with open(path / OPENVINO_FILENAMES["metadata"], "r") as fin:
            metadata = json.load(fin)
        metadata.update(kwargs)
        metadata["network_parameters"] = ModelParams(
            **metadata["network_parameters"]
        )
        model_name = str(path / OPENVINO_FILENAMES["description_file"])
        model_weights = str(path / OPENVINO_FILENAMES["weights"])
        return cls.from_model_name(
            model_name=model_name, model_weights=model_weights, **metadata
        )

    @classmethod
    def from_model_name(
        cls,
        network_parameters: ModelParams,
        model_name: str,
        model_weights: str,
        **kwargs,
    ):
        if len(kwargs) > 0:
            warnings.warn(f"Found extra parameters: {kwargs}")
        inference_engine = IECore()
        network = inference_engine.read_network(
            model=model_name, weights=model_weights
        )
        exec_network = inference_engine.load_network(
            network=network, device_name="CPU"
        )
        input_key = next(iter(exec_network.input_info))
        output_key = next(iter(exec_network.outputs.keys()))
        return cls(
            exec_network,
            input_key,
            output_key,
            network_parameters=network_parameters,
            description_file=model_name,
            weights_file=model_weights,
        )

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        metadata = {
            key: self.__dict__[key] for key in ("input_key", "output_key")
        }
        metadata.update(kwargs)
        return LearnerMetadata.from_model(self, **metadata)

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        metadata = self._get_metadata(**kwargs)
        with open(path / OPENVINO_FILENAMES["metadata"], "w") as fout:
            json.dump(metadata.to_dict(), fout)

        shutil.copy(
            self.description_file,
            path / OPENVINO_FILENAMES["description_file"],
        )
        shutil.copy(self.weights_file, path / OPENVINO_FILENAMES["weights"])

    def _predict_array(self, input_array: np.ndarray) -> np.ndarray:
        result = self.exec_network.infer(inputs={self.input_key: input_array})[
            self.output_key
        ]
        return result


class PytorchOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, PytorchBaseInferenceLearner
):
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_array = input_tensor.cpu().detach().numpy()
        output_array = self._predict_array(input_array)
        return torch.from_numpy(output_array)


class TensorflowOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, TensorflowBaseInferenceLearner
):
    def predict(self, input_tensor: tf.Tensor) -> tf.Tensor:
        input_array = input_tensor.numpy()
        output_array = self._predict_array(input_array)
        return tf.convert_to_tensor(output_array)


OPENVINO_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[OpenVinoInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchOpenVinoInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowOpenVinoInferenceLearner,
}
