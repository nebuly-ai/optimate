import shutil
import warnings
from abc import ABC
import json
from pathlib import Path
from typing import Dict, Union, Type, Generator, Tuple, List

import cpuinfo
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
    if "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower():
        warnings.warn(
            "No valid OpenVino installation has been found. "
            "Trying to re-install it from source."
        )
        from nebullvm.installers.installers import install_openvino

        install_openvino(with_optimization=True)
        from openvino.inference_engine import IECore
    else:
        warnings.warn(
            "No Openvino library detected. "
            "The Openvino Inference learner should not be used."
        )


class OpenVinoInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized using ApacheTVM.

    The class cannot be directly instantiated, but implements all the core
    methods needed for using ApacheTVM at inference time.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def __init__(
        self,
        exec_network,
        input_keys: List,
        output_keys: List,
        description_file: str,
        weights_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.exec_network = exec_network
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.description_file = description_file
        self.weights_file = weights_file

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for the
                `from_model_name` class method.

        Returns:
            OpenVinoInferenceLearner: The optimized model.
        """
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
        """Build the optimized model from the network description and its
        weights.

        Args:
            network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
            model_name (str): File containing a description of the optimized
                model.
            model_weights (str): File containing the model weights.
        """
        if len(kwargs) > 0:
            warnings.warn(f"Found extra parameters: {kwargs}")
        inference_engine = IECore()
        network = inference_engine.read_network(
            model=model_name, weights=model_weights
        )
        exec_network = inference_engine.load_network(
            network=network, device_name="CPU"
        )
        input_keys = list(iter(exec_network.input_info))
        output_keys = list(iter(exec_network.outputs.keys()))
        return cls(
            exec_network,
            input_keys,
            output_keys,
            network_parameters=network_parameters,
            description_file=model_name,
            weights_file=model_weights,
        )

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        # metadata = {
        #     key: self.__dict__[key] for key in ("input_keys", "output_keys")
        # }
        metadata = {}
        metadata.update(kwargs)
        return LearnerMetadata.from_model(self, **metadata)

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        path = Path(path)
        metadata = self._get_metadata(**kwargs)
        with open(path / OPENVINO_FILENAMES["metadata"], "w") as fout:
            json.dump(metadata.to_dict(), fout)

        shutil.copy(
            self.description_file,
            path / OPENVINO_FILENAMES["description_file"],
        )
        shutil.copy(self.weights_file, path / OPENVINO_FILENAMES["weights"])

    def _predict_array(
        self, input_arrays: Generator[np.ndarray, None, None]
    ) -> Generator[np.ndarray, None, None]:
        results = self.exec_network.infer(
            inputs={
                input_key: input_array
                for input_key, input_array in zip(
                    self.input_keys, input_arrays
                )
            }
        )
        return (results[output_key] for output_key in self.output_keys)


class PytorchOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a Pytorch interface.

    This class can be used exactly in the same way as a pytorch Module object.
    At prediction time it takes as input pytorch tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
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
        output_arrays = self._predict_array(input_arrays)
        return tuple(
            torch.from_numpy(output_array) for output_array in output_arrays
        )


class TensorflowOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
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
        output_arrays = self._predict_array(input_arrays)
        return tuple(
            tf.convert_to_tensor(output_array)
            for output_array in output_arrays
        )


OPENVINO_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[OpenVinoInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchOpenVinoInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowOpenVinoInferenceLearner,
}
