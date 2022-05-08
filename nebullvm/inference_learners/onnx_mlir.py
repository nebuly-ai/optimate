import copy
import os
import shutil
import sys
import warnings
from abc import ABC
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Type, Union

import cpuinfo
import numpy as np
import tensorflow as tf
import torch
from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import ONNX_MLIR_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
)

try:
    # Set the ONNX_MLIR_HOME as the environment variable and append in the path,
    # directory path where the MLIR is built
    MLIR_INSTALLATION_ROOT = Path.home()

    os.environ["ONNX_MLIR_HOME"] = os.path.join(
        MLIR_INSTALLATION_ROOT,
        "onnx-mlir",
        "build",
        "Debug",
    )

    sys.path.append(
        os.path.join(
            os.environ.get("ONNX_MLIR_HOME", ""),
            "lib",
        )
    )
    import PyRuntime
except ImportError:
    warnings.warn(
        "No valid mlir-onnx installation found. Trying to install it..."
    )
    from nebullvm.installers.installers import install_onnx_mlir

    install_onnx_mlir(
        working_dir=MLIR_INSTALLATION_ROOT,
    )
    import PyRuntime


class ONNXMlirInferenceLearner(BaseInferenceLearner, ABC):
    """Model converted from ONNX to Shared Object file using ONNX-MLIR dialect
    and run with ONNX-MLIR's PyRuntime
    created at onnx-mlir/build/Debug/lib/PyRuntime.cpython-<target>.so.

    Attributes:
        onnx_mlir_model_path (str or Path): Path to the shared object mlir model.
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
    """

    def __init__(
        self,
        onnx_mlir_model_path: Union[str, Path],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.onnx_mlir_model_path = onnx_mlir_model_path
        self._session = PyRuntime.ExecutionSession(
            os.path.abspath(str(self.onnx_mlir_model_path)),
        )

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)

        shutil.copy(
            self.onnx_mlir_model_path,
            os.path.join(str(path), ONNX_MLIR_FILENAMES["model_name"]),
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
            warnings.warn(
                f"No extra keywords expected for the load method. "
                f"Got {kwargs}."
            )
        onnx_mlir_model_path = os.path.join(
            str(path), ONNX_MLIR_FILENAMES["model_name"]
        )
        metadata = LearnerMetadata.read(path)

        return cls(
            network_parameters=ModelParams(**metadata.network_parameters),
            onnx_mlir_model_path=onnx_mlir_model_path,
        )

    def _predict_arrays(self, input_arrays: Generator[np.ndarray, None, None]):
        outputs = self._session.run(list(input_arrays))
        return outputs


class PytorchONNXMlirInferenceLearner(
    ONNXMlirInferenceLearner, PytorchBaseInferenceLearner
):
    """Model run with ONNX-MLIR's PyRuntime using a Pytorch interface.

    Attributes:
        onnx_mlir_model_path (str or Path): Path to the shared object mlir model.
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
    """

    def predict(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
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
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        outputs = self._predict_arrays(input_arrays)
        return tuple(torch.from_numpy(output) for output in outputs)


class TensorflowONNXMlirInferenceLearner(
    ONNXMlirInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model run with ONNX-MLIR's PyRuntime using a tensorflow interface.

    Attributes:
        onnx_mlir_model_path (str or Path): Path to the shared object mlir model.
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
    """

    def predict(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor]:
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
        input_arrays = (input_tensor.numpy() for input_tensor in input_tensors)
        outputs = self._predict_arrays(input_arrays)
        # noinspection PyTypeChecker
        return tuple(tf.convert_to_tensor(output) for output in outputs)


ONNX_MLIR_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ONNXMlirInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchONNXMlirInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowONNXMlirInferenceLearner,
}
