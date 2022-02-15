import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Type, Dict, Any

import numpy as np
import tensorflow as tf
import torch

from nebullvm.config import TVM_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
)
from nebullvm.base import ModelParams, DeepLearningFramework


try:
    import tvm
    from tvm.contrib.graph_executor import GraphModule
    from tvm.runtime import Module
except ImportError:
    warnings.warn("Not found any valid tvm installation")
    Module = object
    GraphModule = object


@dataclass
class ApacheTVMInferenceLearner(BaseInferenceLearner, ABC):
    graph_executor_module: GraphModule
    input_name: str
    lib: Module
    target: str

    def _predict_array(self, input_array: np.ndarray):
        self.graph_executor_module.set_input(self.input_name, input_array)
        self.graph_executor_module.run()
        output_shape = (
            self.network_parameters.batch_size,
            *self.network_parameters.output_size,
        )
        tvm_output = self.graph_executor_module.get_output(
            0, tvm.nd.empty(output_shape)
        ).numpy()
        return tvm_output

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.from_model(
            self, input_name=self.input_name, target=self.target, **kwargs
        )
        metadata.save(path)
        self.lib.export_library(path / TVM_FILENAMES["engine"])

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.read(path).to_dict()
        network_parameters = ModelParams(**metadata["network_parameters"])
        lib = tvm.runtime.load_module(path / TVM_FILENAMES["engine"])
        target_device = metadata["target"]
        input_name = metadata["input_name"]
        return cls.from_runtime_module(
            network_parameters=network_parameters,
            lib=lib,
            target_device=target_device,
            input_name=input_name,
        )

    @classmethod
    def from_runtime_module(
        cls,
        network_parameters: ModelParams,
        lib: Module,
        target_device: str,
        input_name: str,
    ):
        dev = tvm.device(str(target_device), 0)
        graph_executor_module = GraphModule(lib["default"](dev))
        return cls(
            network_parameters=network_parameters,
            graph_executor_module=graph_executor_module,
            input_name=input_name,
            lib=lib,
            target=target_device,
        )


class PytorchApacheTVMInferenceLearner(
    ApacheTVMInferenceLearner, PytorchBaseInferenceLearner
):
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        device = self._convert_device(input_tensor.get_device())
        input_array = input_tensor.cpu().detach().numpy()
        output_array = self._predict_array(input_array)
        return torch.from_numpy(output_array).to(device)

    @staticmethod
    def _convert_device(device: Any):
        if isinstance(device, int):
            return "cpu"
        return device


class TensorflowApacheTVMInferenceLearner(
    ApacheTVMInferenceLearner, TensorflowBaseInferenceLearner
):
    def predict(self, input_tensor: tf.Tensor):
        input_array = input_tensor.numpy()
        output_array = self._predict_array(input_array)
        return tf.convert_to_tensor(output_array)


TVM_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ApacheTVMInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchApacheTVMInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowApacheTVMInferenceLearner,
}
