import json
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, Dict, Type

import tensorflow as tf
import torch

from nebullvm.config import NVIDIA_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
)
from nebullvm.base import ModelParams, DeepLearningFramework

if torch.cuda.is_available():
    import tensorrt as trt
    import polygraphy


@dataclass
class NvidiaInferenceLearner(BaseInferenceLearner, ABC):
    engine: Any
    input_name: str
    output_name: str
    cuda_stream: Any = None
    nvidia_logger: Any = None

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        metadata = {
            key: self.__dict__[key] for key in ("input_name", "output_name")
        }
        metadata.update(kwargs)
        return LearnerMetadata.from_model(self, **metadata)

    def _synchronize_stream(self):
        raise NotImplementedError()

    @property
    def stream_ptr(self):
        raise NotImplementedError()

    @staticmethod
    def _get_default_cuda_stream() -> Any:
        raise NotImplementedError()

    @staticmethod
    def check_env():
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )

    def __post_init__(self):
        self.check_env()
        if self.nvidia_logger is None:
            self.nvidia_logger = trt.Logger(trt.Logger.WARNING)
        if self.cuda_stream is None:
            self.cuda_stream = self._get_default_cuda_stream()

    @classmethod
    def from_engine_path(
        cls,
        network_parameters: ModelParams,
        engine_path: Union[str, Path],
        input_name: str,
        output_name: str,
        nvidia_logger: Any = None,
        cuda_stream: Any = None,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"Debug: Got extra keywords in "
                f"NvidiaInferenceLearner::from_engine_path: {kwargs}"
            )
        if nvidia_logger is None:
            nvidia_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(nvidia_logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return cls(
            network_parameters=network_parameters,
            engine=engine,
            input_name=input_name,
            output_name=output_name,
            nvidia_logger=nvidia_logger,
            cuda_stream=cuda_stream,
        )

    def _predict_tensor(self, input_ptr: Any, output_ptr: Any):
        context = self.engine.create_execution_context()
        input_idx = self.engine[self.input_name]
        output_idx = self.engine[self.output_name]
        buffers = [None] * 2  # Assuming 1 input and 1 output
        buffers[input_idx] = input_ptr
        buffers[output_idx] = output_ptr
        context.execute_async_v2(buffers, self.stream_ptr)
        self._synchronize_stream()

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        serialized_engine = self.engine.serialize()
        with open(path / NVIDIA_FILENAMES["engine"], "wb") as fout:
            fout.write(serialized_engine)
        metadata = self._get_metadata(**kwargs)
        with open(path / NVIDIA_FILENAMES["metadata"], "w") as fout:
            json.dump(metadata.to_dict(), fout)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        with open(path / NVIDIA_FILENAMES["metadata"], "r") as fin:
            metadata = json.load(fin)
        metadata.update(kwargs)
        metadata["network_parameters"] = ModelParams(
            **metadata["network_parameters"]
        )
        return cls.from_engine_path(
            path / NVIDIA_FILENAMES["engine"], **metadata
        )


class PytorchNvidiaInferenceLearner(
    NvidiaInferenceLearner, PytorchBaseInferenceLearner
):
    def _synchronize_stream(self):
        self.cuda_stream.synchronize()

    @staticmethod
    def _get_default_cuda_stream() -> Any:
        return torch.cuda.default_stream()

    @property
    def stream_ptr(self):
        return self.cuda_stream.cuda_stream

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.cuda()
        output_size = (
            self.network_parameters.batch_size,
            *self.network_parameters.output_size,
        )
        output_tensor = torch.Tensor(*output_size).cuda()
        input_ptr = input_tensor.data_ptr()
        output_ptr = output_tensor.data_ptr()
        self._predict_tensor(input_ptr, output_ptr)
        return output_tensor.cpu()


class TensorflowNvidiaInferenceLearner(
    NvidiaInferenceLearner, TensorflowBaseInferenceLearner
):
    def _synchronize_stream(self):
        self.cuda_stream.synchronize()

    @staticmethod
    def _get_default_cuda_stream() -> Any:
        return polygraphy.Stream()

    @property
    def stream_ptr(self):
        return self.cuda_stream.ptr

    def predict(self, input_tensor: tf.Tensor) -> tf.Tensor:
        output_size = (
            self.network_parameters.batch_size,
            *self.network_parameters.output_size,
        )
        input_array = input_tensor.numpy()
        cuda_input_array = polygraphy.DeviceArray.copy_from(
            input_array, stream=self.cuda_stream
        )
        cuda_output_array = polygraphy.DeviceArray(shape=output_size)
        input_ptr = cuda_input_array.ptr
        output_ptr = cuda_output_array.ptr
        self._predict_tensor(input_ptr, output_ptr)
        output_array = cuda_output_array.numpy()
        cuda_input_array.free()
        cuda_output_array.free()
        return output_array


NVIDIA_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[NvidiaInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchNvidiaInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowNvidiaInferenceLearner,
}
