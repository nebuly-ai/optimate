import os
import shutil
import warnings
from abc import ABC
from pathlib import Path
from typing import Union, List, Generator, Tuple, Dict, Type

import numpy as np
import torch

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import ONNX_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
)
from nebullvm.installers.installers import install_deepsparse
from nebullvm.transformations.base import MultiStageTransformation

try:
    from deepsparse import compile_model, cpu
except ImportError:
    import platform

    os_ = platform.system()
    if os_ != "Darwin":
        warnings.warn(
            "No deepsparse installation found. Trying to install it..."
        )
        install_deepsparse()
        from deepsparse import compile_model, cpu
    else:
        warnings.warn(
            "No valid deepsparse installation found. "
            "The compiler won't be used in the following."
        )


class DeepSparseInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized on CPU using DeepSparse. DeepSparse is an engine
    accelerating sparse computations on CPUs.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        input_names: List[str],
        output_names: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.onnx_path = self._store_file(onnx_path)

        # Compile model
        cores_per_socket, _, _ = cpu.cpu_details()
        # Define the number of cores to use, by default it will make use of
        # all physical cores on the system
        num_cores = cores_per_socket
        batch_size = kwargs["network_parameters"].batch_size
        self.engine = compile_model(onnx_path, batch_size, num_cores)

        self.input_names = input_names
        self.output_names = output_names

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
        metadata.save(path)

        shutil.copy(
            self.onnx_path,
            Path(path) / ONNX_FILENAMES["model_name"],
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
            DeepSparseInferenceLearner: The optimized model.
        """
        if len(kwargs) > 0:
            warnings.warn(
                f"No extra keywords expected for the load method. "
                f"Got {kwargs}."
            )
        onnx_path = os.path.join(str(path), ONNX_FILENAMES["model_name"])
        metadata = LearnerMetadata.read(path)
        input_tfms = metadata.input_tfms
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
        )

    def _predict_arrays(self, input_arrays: Generator[np.ndarray, None, None]):
        inputs = [array for array in input_arrays]
        outputs = self.engine(inputs)
        return outputs


class PytorchDeepSparseInferenceLearner(
    DeepSparseInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized on CPU using DeepSparse. DeepSparse is an engine
    accelerating sparse computations on CPUs.

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
        input_arrays = (
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        outputs = self._predict_arrays(input_arrays)
        return tuple(torch.from_numpy(output) for output in outputs)


DEEPSPARSE_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[DeepSparseInferenceLearner]
] = {DeepLearningFramework.PYTORCH: PytorchDeepSparseInferenceLearner}
