import json
import os
from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Union, Dict, Type, List, Tuple, Generator, Optional

import numpy as np
from loguru import logger

from nebullvm.config import NVIDIA_FILENAMES
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
    NumpyBaseInferenceLearner,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.tensor_rt import tensorrt as trt, polygraphy
from nebullvm.optional_modules.torch import torch, ScriptModule
from nebullvm.tools.base import (
    DeviceType,
    ModelParams,
    DeepLearningFramework,
    Device,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    VerifyContiguity,
)


class ONNXTensorRTInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized using TensorRT.

    The class cannot be directly instantiated, but implements all the core
    methods needed for using TensorRT at inference time.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        engine (any): The tensorRT engine.
        input_names (List[str]): Names associated to the model input tensors.
        output_names (List[str]): Names associated to the model output tensors.
        cuda_stream (any, optional): Stream used for communication with Nvidia
            GPUs.
        nvidia_logger (any, optional): Logger used by the Nvidia service
    """

    name = "TensorRT"

    def __init__(
        self,
        engine: Any,
        input_names: List[str],
        output_names: List[str],
        device: Device,
        cuda_stream: Any = None,
        nvidia_logger: Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names
        self.cuda_stream = cuda_stream
        self.nvidia_logger = nvidia_logger
        self.output_tensors = None
        self.device = device
        self._set_cuda_env(device.type is DeviceType.GPU)

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        metadata = {
            key: self.__dict__[key] for key in ("input_names", "output_names")
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
    def check_env(use_gpu):
        if not use_gpu:
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )

    def _set_cuda_env(self, use_gpu):
        self.check_env(use_gpu)
        if self.nvidia_logger is None:
            self.nvidia_logger = trt.Logger(trt.Logger.WARNING)
        if self.cuda_stream is None:
            self.cuda_stream = self._get_default_cuda_stream()

    @classmethod
    def from_engine_path(
        cls,
        network_parameters: ModelParams,
        engine_path: Union[str, Path],
        input_names: List[str],
        output_names: List[str],
        device: Device,
        nvidia_logger: Any = None,
        cuda_stream: Any = None,
        input_tfms: MultiStageTransformation = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Build the model from the serialised engine.

        Args:
            network_parameters (ModelParams): Model parameters.
            engine_path (str or Path): Path to the serialised engine. The
                serialised engine is the serialised version of the engine
                used for accelerating the inference.
            input_names (List[str]): Names associated to the model input
                tensors.
            output_names (List[str]): Names associated to the model output
                tensors.
            device: (Device): Device where the model wil be run.
            cuda_stream (any, optional): Stream used for communication with
                Nvidia GPUs.
            nvidia_logger (any, optional): Logger used by the Nvidia service
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            NvidiaInferenceLearner: The optimized model.
        """
        if kwargs:
            logger.warning(
                f"Debug: Got extra keywords in "
                f"NvidiaInferenceLearner::from_engine_path: {kwargs}"
            )
        if nvidia_logger is None:
            nvidia_logger = trt.Logger(trt.Logger.WARNING)
        if input_tfms is None:
            input_tfms = MultiStageTransformation([])
        input_tfms.append(VerifyContiguity())
        runtime = trt.Runtime(nvidia_logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return cls(
            input_tfms=input_tfms,
            network_parameters=network_parameters,
            engine=engine,
            input_names=input_names,
            output_names=output_names,
            nvidia_logger=nvidia_logger,
            cuda_stream=cuda_stream,
            input_data=input_data,
            device=device,
        )

    def _predict_tensors(
        self,
        input_ptrs: Generator[Any, None, None],
        output_ptrs: Generator[Any, None, None],
        input_shapes: Generator[Any, None, None] = None,
    ):
        buffers = [None] * (len(self.input_names) + len(self.output_names))
        input_idxs = (
            self.engine[input_name] for input_name in self.input_names
        )
        output_idxs = (
            self.engine[output_name] for output_name in self.output_names
        )
        input_shapes = input_shapes or [None] * len(self.input_names)
        for input_idx, input_ptr, input_shape in zip(
            input_idxs, input_ptrs, input_shapes
        ):
            buffers[input_idx] = input_ptr
            if input_shape is not None:
                self.context.set_binding_shape(input_idx, input_shape)
        for output_idx, output_ptr in zip(output_idxs, output_ptrs):
            buffers[output_idx] = output_ptr
        self.context.execute_async_v2(buffers, self.stream_ptr)
        self._synchronize_stream()

    def get_size(self):
        return self.engine.serialize().nbytes

    def free_gpu_memory(self):
        # ONNXtensorrt doesn't need to release gpu memory
        pass

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        serialized_engine = self.engine.serialize()
        with open(path / NVIDIA_FILENAMES["engine"], "wb") as fout:
            fout.write(serialized_engine)
        metadata = self._get_metadata(**kwargs)
        with open(path / NVIDIA_FILENAMES["metadata"], "w") as fout:
            json.dump(metadata.to_dict(), fout)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for the
                `from_engine_path` class method.

        Returns:
            ONNXTensorRTInferenceLearner: The optimized model.
        """
        path = Path(path)
        with open(path / NVIDIA_FILENAMES["metadata"], "r") as fin:
            metadata = json.load(fin)
        metadata.update(kwargs)
        metadata["network_parameters"] = ModelParams(
            **metadata["network_parameters"]
        )
        input_tfms = metadata.get("input_tfms")
        if input_tfms is not None:
            metadata["input_tfms"] = MultiStageTransformation.from_dict(
                input_tfms
            )
        metadata["device"] = Device(DeviceType.GPU)
        return cls.from_engine_path(
            engine_path=path / NVIDIA_FILENAMES["engine"],
            **metadata,
        )


class PytorchTensorRTInferenceLearner(PytorchBaseInferenceLearner):
    MODEL_NAME = "model_optimized.pt"
    name = "TensorRT"

    def __init__(
        self,
        torch_model: ScriptModule,
        device: Device,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = torch_model.eval()
        if device.type is DeviceType.GPU:
            self.model.to(device.to_torch_format())
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.device = device
        self._is_gpu_ready = device.type is DeviceType.GPU

    def get_size(self):
        with TemporaryDirectory() as tmp_dir:
            self.save(tmp_dir)
            return sum(
                os.path.getsize(Path(tmp_dir) / f)
                for f in os.listdir(Path(tmp_dir))
                if os.path.isfile(Path(tmp_dir) / f)
            )

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.device.type is DeviceType.GPU and not self._is_gpu_ready:
            self.set_model_on_gpu()

        # PyTorch-TensorRT does not support int64
        input_tensors = (
            t.to(self.device.to_torch_format())
            if t.dtype != torch.int64
            else t.to(torch.int32).to(self.device.to_torch_format())
            for t in input_tensors
        )

        with torch.no_grad():
            res = self.model(*input_tensors)
            if not isinstance(res, tuple):
                res = res.to(self.device.to_torch_format())
                return (res,)
            return tuple(out.to(self.device.to_torch_format()) for out in res)

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        torch.jit.save(self.model, path / self.MODEL_NAME)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        model = torch.jit.load(path / cls.MODEL_NAME)
        metadata = LearnerMetadata.read(path)
        device = Device(DeviceType.GPU)
        return cls(
            torch_model=model,
            network_parameters=ModelParams(**metadata.network_parameters),
            input_tfms=MultiStageTransformation.from_dict(metadata.input_tfms)
            if metadata.input_tfms is not None
            else None,
            device=device,
        )


class PytorchONNXTensorRTInferenceLearner(
    ONNXTensorRTInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized using TensorRT with a Pytorch interface.

    This class can be used exactly in the same way as a pytorch Module object.
    At prediction time it takes as input pytorch tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        engine (any): The tensorRT engine.
        input_names (List[str]): Names associated to the model input tensors.
        output_names (List[str]): Names associated to the model output tensors.
        cuda_stream (any, optional): Stream used for communication with Nvidia
            GPUs.
        nvidia_logger (any, optional): Logger used by the Nvidia service.
    """

    def _synchronize_stream(self):
        self.cuda_stream.synchronize()

    @staticmethod
    def _get_default_cuda_stream() -> Any:
        return torch.cuda.default_stream()

    @property
    def stream_ptr(self):
        return self.cuda_stream.cuda_stream

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_tensors = [
            input_tensor.to(self.device.to_torch_format())
            for input_tensor in input_tensors
        ]
        if self.network_parameters.dynamic_info is None:
            if self.output_tensors is None:
                self.output_tensors = [
                    torch.Tensor(*output_size).to(
                        self.device.to_torch_format()
                    )
                    for output_size in self.network_parameters.output_sizes
                ]
            input_sizes = None
        else:
            dynamic_info = self.network_parameters.dynamic_info
            input_sizes = [
                input_tensor.size() for input_tensor in input_tensors
            ]
            self.output_tensors = [
                torch.Tensor(
                    *(
                        x
                        if i not in dynamic_axis.keys()
                        else dynamic_info.retrieve_output_dim(
                            input_sizes, j, i, x
                        )
                        for i, x in enumerate(output_size)
                    ),
                ).to(self.device.to_torch_format())
                for j, (output_size, dynamic_axis) in enumerate(
                    zip(
                        self.network_parameters.output_sizes,
                        dynamic_info.outputs,
                    )
                )
            ]
        input_ptrs = (
            input_tensor.data_ptr() for input_tensor in input_tensors
        )
        output_ptrs = (
            output_tensor.data_ptr() for output_tensor in self.output_tensors
        )
        self._predict_tensors(input_ptrs, output_ptrs, input_sizes)
        return tuple(
            output_tensor.to(self.device.to_torch_format())
            for output_tensor in self.output_tensors
        )


class BaseArrayONNXTensorRTInferenceLearner(ONNXTensorRTInferenceLearner, ABC):
    """Base Model that can be used for all array-based
    NvidiaInferenceLearners.
    """

    def _synchronize_stream(self):
        self.cuda_stream.synchronize()

    @staticmethod
    def _get_default_cuda_stream() -> Any:
        return polygraphy.cuda.Stream()

    @property
    def stream_ptr(self):
        return self.cuda_stream.ptr

    @staticmethod
    def _convert_to_array_and_free_memory(cuda_array) -> np.ndarray:
        array = cuda_array.numpy()
        cuda_array.free()
        return array

    def _predict_array(
        self,
        cuda_input_arrays: List,
        input_shapes: Optional[List[Tuple[int, ...]]],
    ) -> Generator[np.ndarray, None, None]:
        if self.network_parameters.dynamic_info is None:
            cuda_output_arrays = [
                polygraphy.cuda.DeviceArray(shape=output_size)
                for output_size in self.network_parameters.output_sizes
            ]
        else:
            dynamic_info = self.network_parameters.dynamic_info
            output_sizes = (
                output_size
                for output_size in self.network_parameters.output_sizes
            )

            cuda_output_arrays = [
                polygraphy.cuda.DeviceArray(
                    shape=tuple(
                        x
                        if i not in dyn_out_axis.keys()
                        else dynamic_info.retrieve_output_dim(
                            input_shapes, j, i, x
                        )
                        for i, x in enumerate(output_size)
                    )
                )
                for j, (output_size, dyn_out_axis) in enumerate(
                    zip(output_sizes, dynamic_info.outputs)
                )
            ]
        input_ptrs = (cuda_array.ptr for cuda_array in cuda_input_arrays)
        output_ptrs = (cuda_array.ptr for cuda_array in cuda_output_arrays)
        self._predict_tensors(input_ptrs, output_ptrs, input_shapes)
        for cuda_input_array in cuda_input_arrays:
            cuda_input_array.free()
        return (
            self._convert_to_array_and_free_memory(array)
            for array in cuda_output_arrays
        )


class TensorflowONNXTensorRTInferenceLearner(
    BaseArrayONNXTensorRTInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model optimized using TensorRT with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        engine (any): The tensorRT engine.
        input_names (List[str]): Names associated to the model input tensors.
        output_names (List[str]): Names associated to the model output tensors.
        cuda_stream (any, optional): Stream used for communication with Nvidia
            GPUs.
        nvidia_logger (any, optional): Logger used by the Nvidia service.
    """

    def run(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        cuda_input_arrays = [
            polygraphy.cuda.DeviceArray(
                shape=tuple(input_tensor.shape),
                dtype=input_tensor.numpy().dtype,
            ).copy_from(input_tensor.numpy(), stream=self.cuda_stream)
            for input_tensor in input_tensors
        ]
        input_shapes = (
            [tuple(input_tensor.shape) for input_tensor in input_tensors]
            if self.network_parameters.dynamic_info is not None
            else None
        )
        out_arrays = self._predict_array(cuda_input_arrays, input_shapes)
        return tuple(tf.convert_to_tensor(array) for array in out_arrays)


class NumpyONNXTensorRTInferenceLearner(
    BaseArrayONNXTensorRTInferenceLearner, NumpyBaseInferenceLearner
):
    """Model optimized using TensorRT with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        engine (any): The tensorRT engine.
        input_names (List[str]): Names associated to the model input tensors.
        output_names (List[str]): Names associated to the model output tensors.
        cuda_stream (any, optional): Stream used for communication with Nvidia
            GPUs.
        nvidia_logger (any, optional): Logger used by the Nvidia service.
    """

    def run(self, *input_tensors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[np.ndarray]): Input tensors belonging to
                the same batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[np.ndarray]: Output tensors. Note that the output tensors
                does not correspond to the prediction on the input tensors
                with a 1 to 1 mapping. In fact the output tensors are produced
                as the multiple-output of the model given a (multi-) tensor
                input.
        """
        cuda_input_arrays = [
            polygraphy.cuda.DeviceArray(
                shape=tuple(input_tensor.shape), dtype=input_tensor.dtype
            ).copy_from(input_tensor, stream=self.cuda_stream)
            for input_tensor in input_tensors
        ]
        input_shapes = (
            [tuple(input_tensor.shape) for input_tensor in input_tensors]
            if self.network_parameters.dynamic_info is not None
            else None
        )
        return tuple(self._predict_array(cuda_input_arrays, input_shapes))


TENSOR_RT_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ONNXTensorRTInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchONNXTensorRTInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowONNXTensorRTInferenceLearner,
    DeepLearningFramework.NUMPY: NumpyONNXTensorRTInferenceLearner,
}
