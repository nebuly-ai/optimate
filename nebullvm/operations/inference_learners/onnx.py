import multiprocessing
import os
import shutil
from abc import ABC
from pathlib import Path
from typing import Union, List, Generator, Tuple, Dict, Type

import cpuinfo
import numpy as np
from loguru import logger

from nebullvm.config import (
    ONNX_FILENAMES,
    ONNX_PROVIDERS,
)
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
    NumpyBaseInferenceLearner,
)
from nebullvm.optional_modules.onnx import onnx
from nebullvm.optional_modules.onnxruntime import onnxruntime as ort
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import DeepLearningFramework, Device, ModelParams
from nebullvm.tools.transformations import MultiStageTransformation


def _running_on_intel_cpu(use_gpu):
    if use_gpu:
        return False  # running on GPU
    cpu_info = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" in cpu_info:
        return True
    return False


def _get_ort_session_options(use_gpu) -> ort.SessionOptions:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if not use_gpu:
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = max(
            int(
                os.environ.get("NEBULLVM_THREADS_PER_MODEL")
                or multiprocessing.cpu_count()
            ),
            1,
        )
    return sess_options


class ONNXInferenceLearner(BaseInferenceLearner, ABC):
    """Model converted to ONNX and run with Microsoft's onnxruntime.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    name = "ONNXRuntime"

    def __init__(
        self,
        onnx_path: Union[str, Path],
        input_names: List[str],
        output_names: List[str],
        device: Device,
        **kwargs,
    ):
        super().__init__(**kwargs)
        filename = Path(onnx_path).name
        dir_path = str(Path(onnx_path).parent)
        self.device = device

        self.onnx_path = Path(self._store_dir(dir_path)) / filename
        sess_options = _get_ort_session_options(self.device is Device.GPU)

        if _running_on_intel_cpu(self.device is Device.GPU):
            sess_options.add_session_config_entry(
                "session.set_denormal_as_zero", "1"
            )

        ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=ONNX_PROVIDERS["cuda"]
            if self.device is Device.GPU
            else ONNX_PROVIDERS["cpu"],
        )
        self._session = ort_session
        self.input_names = input_names
        self.output_names = output_names

    def get_size(self):
        return sum(
            os.path.getsize(self.onnx_path.parents[0] / f)
            for f in os.listdir(self.onnx_path.parents[0])
            if os.path.isfile(self.onnx_path.parents[0] / f)
        )

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        metadata = LearnerMetadata.from_model(
            self,
            input_names=self.input_names,
            output_names=self.output_names,
            **kwargs,
        )

        path = Path(path)
        path.mkdir(exist_ok=True)

        metadata.save(path)

        shutil.copy(
            self.onnx_path,
            os.path.join(str(path), ONNX_FILENAMES["model_name"]),
        )

        try:
            # Tries to load the model
            onnx.load(os.path.join(str(path), ONNX_FILENAMES["model_name"]))
        except FileNotFoundError:
            # If missing files, it means it's saved in onnx external_data
            # format
            src_dir = str(Path(self.onnx_path).parent)
            files = os.listdir(src_dir)
            for fname in files:
                if ".onnx" not in fname:
                    shutil.copy2(
                        os.path.join(src_dir, fname), os.path.join(path, fname)
                    )

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for consistency
                with other Learners.

        Returns:
            ONNXInferenceLearner: The optimized model.
        """
        if len(kwargs) > 0:
            logger.warning(
                f"No extra keywords expected for the load method. "
                f"Got {kwargs}."
            )
        path = Path(path)
        onnx_path = path / ONNX_FILENAMES["model_name"]
        metadata = LearnerMetadata.read(path)
        input_tfms = metadata.input_tfms
        device = metadata.device
        if input_tfms is not None:
            input_tfms = MultiStageTransformation.from_dict(
                metadata.input_tfms
            )
        return cls(
            input_tfms=input_tfms,
            network_parameters=ModelParams(**metadata.network_parameters),
            onnx_path=onnx_path,
            input_names=metadata["input_names"],
            output_names=metadata["output_names"],
            device=device,
        )

    def _predict_arrays(self, input_arrays: Generator[np.ndarray, None, None]):
        input_dict = {
            name: input_array
            for name, input_array in zip(self.input_names, input_arrays)
        }
        outputs = self._session.run(self.output_names, input_dict)
        return outputs


class PytorchONNXInferenceLearner(
    ONNXInferenceLearner, PytorchBaseInferenceLearner
):
    """Model run with Microsoft's onnxruntime using a Pytorch interface.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

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
        device = input_tensors[0].device
        input_arrays = (
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        outputs = self._predict_arrays(input_arrays)
        return tuple(torch.from_numpy(output).to(device) for output in outputs)


class TensorflowONNXInferenceLearner(
    ONNXInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model run with Microsoft's onnxruntime using a tensorflow interface.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
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
        input_arrays = (
            input_tensor.numpy()
            if not isinstance(input_tensor, np.ndarray)
            else input_tensor
            for input_tensor in input_tensors
        )
        outputs = self._predict_arrays(input_arrays)
        # noinspection PyTypeChecker
        return tuple(tf.convert_to_tensor(output) for output in outputs)


class NumpyONNXInferenceLearner(
    ONNXInferenceLearner, NumpyBaseInferenceLearner
):
    """Model run with Microsoft's onnxruntime using a numpy interface.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    def run(self, *input_tensors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[np.ndarray, ...]): Input tensors belonging to
                the same batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (input_tensor for input_tensor in input_tensors)
        outputs = self._predict_arrays(input_arrays)
        return tuple(outputs)


ONNX_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ONNXInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchONNXInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowONNXInferenceLearner,
    DeepLearningFramework.NUMPY: NumpyONNXInferenceLearner,
}
