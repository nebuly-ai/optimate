import shutil
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Type, Dict, Any, List, Generator, Tuple

import numpy as np
import tensorflow as tf
import torch

from nebullvm.config import TVM_FILENAMES, NO_COMPILER_INSTALLATION
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
    if not NO_COMPILER_INSTALLATION:
        warnings.warn(
            "Not found any valid tvm installation. "
            "Trying to install it from source."
        )
        from nebullvm.installers.installers import install_tvm

        install_tvm()
        import tvm
        from tvm.contrib.graph_executor import GraphModule
        from tvm.runtime import Module
    else:
        warnings.warn(
            "Not found any valid tvm installation. "
            "TVM will not be available in the following steps."
        )
        Module = object
        GraphModule = object


@dataclass
class ApacheTVMInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized using ApacheTVM.

    The class cannot be directly instantiated, but implements all the core
    methods needed for using ApacheTVM at inference time.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        graph_executor_module (GraphModule): The graph executor. This is the
            central component in the ApacheTVM optimized model execution.
        input_names (List[str]): Names associated to the model input tensors.
        lib (Module): Component needed for loading the ApacheTVM optimized
            model.
        target (str): Target device. It can be wither `llvm` for targeting CPUs
            or "cuda" for targeting GPUs.
        engine_path (Path, optional): Path to the serialized engine. To be used
            after loading the model (avoiding double engine serialization).
    """

    graph_executor_module: GraphModule
    input_names: List[str]
    lib: Module
    target: str
    engine_path: Path = None

    def _predict_array(
        self, input_arrays: Generator[np.ndarray, None, None]
    ) -> Generator[np.ndarray, None, None]:
        for name, array in zip(self.input_names, input_arrays):
            self.graph_executor_module.set_input(name, array)
        self.graph_executor_module.run()

        tvm_outputs = (
            self.graph_executor_module.get_output(
                i,
                tvm.nd.empty(
                    (
                        self.network_parameters.batch_size,
                        *output_size,
                    )
                ),
            ).numpy()
            for i, output_size in enumerate(
                self.network_parameters.output_sizes
            )
        )
        return tvm_outputs

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        path = Path(path)
        metadata = LearnerMetadata.from_model(
            self, input_names=self.input_names, target=self.target, **kwargs
        )
        metadata.save(path)
        if self.engine_path is None:
            self.lib.export_library(path / TVM_FILENAMES["engine"])
        else:
            shutil.copy(self.engine_path, path)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for the
                `from_runtime_module` class method.

        Returns:
            ApacheTVMInferenceLearner: The optimized model.
        """
        path = Path(path)
        metadata = LearnerMetadata.read(path).to_dict()
        network_parameters = ModelParams(**metadata["network_parameters"])
        lib = tvm.runtime.load_module(path / TVM_FILENAMES["engine"])
        target_device = metadata["target"]
        input_names = metadata["input_names"]
        self = cls.from_runtime_module(
            network_parameters=network_parameters,
            lib=lib,
            target_device=target_device,
            input_names=input_names,
        )
        self.engine_path = path / TVM_FILENAMES["engine"]
        return self

    @classmethod
    def from_runtime_module(
        cls,
        network_parameters: ModelParams,
        lib: Module,
        target_device: str,
        input_names: List[str],
    ):
        """Build the model from the runtime module (lib).

        Args:
            network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
            lib (Module): Component needed for loading the ApacheTVM optimized
                model.
            target_device (str): The target device. Either `llvm` (CPU)
                or `cuda`.
            input_names (List[str]): Names associated to the model input
                tensors.
        """
        dev = tvm.device(str(target_device), 0)
        graph_executor_module = GraphModule(lib["default"](dev))
        return cls(
            network_parameters=network_parameters,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=lib,
            target=target_device,
        )


class PytorchApacheTVMInferenceLearner(
    ApacheTVMInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a Pytorch interface.

    This class can be used exactly in the same way as a pytorch Module object.
    At prediction time it takes as input pytorch tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        graph_executor_module (GraphModule): The graph executor. This is the
            central component in the ApacheTVM optimized model execution.
        input_names (List[str]): Names associated to the model input tensors.
        lib (Module): Component needed for loading the ApacheTVM optimized
            model.
        target (str): Target device. It can be wither `llvm` for targeting CPUs
            or "cuda" for targeting GPUs.
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
        device = self._convert_device(input_tensors[0].get_device())
        input_arrays = (
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        output_arrays = self._predict_array(input_arrays)
        return tuple(
            torch.from_numpy(output_array).to(device)
            for output_array in output_arrays
        )

    @staticmethod
    def _convert_device(device: Any):
        if isinstance(device, int):
            return "cpu"
        return device


class TensorflowApacheTVMInferenceLearner(
    ApacheTVMInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        graph_executor_module (GraphModule): The graph executor. This is the
            central component in the ApacheTVM optimized model execution.
        input_names (List[str]): Names associated to the model input tensors.
        lib (Module): Component needed for loading the ApacheTVM optimized
            model.
        target (str): Target device. It can be wither `llvm` for targeting CPUs
            or "cuda" for targeting GPUs.
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


TVM_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ApacheTVMInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchApacheTVMInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowApacheTVMInferenceLearner,
}
