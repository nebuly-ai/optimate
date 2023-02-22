import os
import shutil
from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, Type, Dict, Any, List, Generator, Tuple, Optional

import numpy as np

from nebullvm.config import (
    TVM_FILENAMES,
)
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
    NumpyBaseInferenceLearner,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.optional_modules.tvm import (
    GraphModule,
    tvm,
    ExecutorFactoryModule,
)
from nebullvm.tools.base import (
    ModelParams,
    DeepLearningFramework,
    Device,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    HalfPrecisionTransformation,
)


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

    name = "ApacheTVM"

    def __init__(
        self,
        graph_executor_module: GraphModule,
        input_names: List[str],
        lib: ExecutorFactoryModule,
        target: str,
        device: Device,
        engine_path: Path = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.graph_executor_module = graph_executor_module
        self.input_names = input_names
        self.lib = lib
        self.target = target
        self.engine_path = (
            self._store_file(engine_path)
            if engine_path is not None
            else engine_path
        )
        self.device = device

    def get_size(self):
        with TemporaryDirectory() as tmp_dir:
            self.save(tmp_dir)
            return sum(
                os.path.getsize(Path(tmp_dir) / f)
                for f in os.listdir(Path(tmp_dir))
                if os.path.isfile(Path(tmp_dir) / f)
            )

    def _has_half_precision_transformation(self):
        for tfm in self.input_tfms.to_list():
            if isinstance(tfm, HalfPrecisionTransformation):
                return True
        return False

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
                    shape=output_size,
                    dtype="float16"
                    if self._has_half_precision_transformation()
                    else "float32",
                ),
            ).numpy()
            for i, output_size in enumerate(
                self.network_parameters.output_sizes
            )
        )
        return tvm_outputs

    def free_gpu_memory(self):
        # TODO: check if tvm needs to release GPU
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
        input_tfms = metadata.get("input_tfms")
        if input_tfms is not None:
            metadata["input_tfms"] = MultiStageTransformation.from_dict(
                input_tfms
            )
        device = Device.from_str(metadata["device"])
        self = cls.from_runtime_module(
            network_parameters=network_parameters,
            lib=lib,
            target_device=target_device,
            input_names=input_names,
            device=device,
        )
        self.engine_path = path / TVM_FILENAMES["engine"]
        return self

    @classmethod
    def from_runtime_module(
        cls,
        network_parameters: ModelParams,
        lib: ExecutorFactoryModule,
        target_device: str,
        input_names: List[str],
        device: Device,
        input_tfms: MultiStageTransformation = None,
        input_data: DataManager = None,
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
            device (Device): The device where the model will be executed.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            input_data (DataManager, optional): User defined data.
        """
        dev = tvm.device(str(target_device), 0)
        graph_executor_module = GraphModule(lib["default"](dev))
        return cls(
            input_tfms=input_tfms,
            network_parameters=network_parameters,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=lib,
            target=target_device,
            input_data=input_data,
            device=device,
        )


class BaseArrayApacheTVMInferenceLearner(ApacheTVMInferenceLearner, ABC):
    """Base Model that can be used for all array-based
    ApacheTVMInferenceLearners.
    """

    def _inner_predict(
        self,
        input_arrays: Generator[np.ndarray, None, None],
        input_shapes: Optional[List[Tuple[int, ...]]],
    ) -> Generator[np.ndarray, None, None]:
        if self.network_parameters.dynamic_info is not None:
            input_arrays = (
                np.pad(
                    input_array,
                    [
                        (0, abs(x - y))
                        for x, y in zip(
                            input_array.shape,
                            input_size,
                        )
                    ],
                    mode="constant",
                    constant_values=0,
                )
                for input_array, input_size in zip(
                    input_arrays, self.network_parameters.input_sizes
                )
            )

        output_arrays = self._predict_array(input_arrays)
        if self.network_parameters.dynamic_info is not None:
            assert input_shapes is not None
            dynamic_info = self.network_parameters.dynamic_info
            return (
                output_array[
                    tuple(
                        slice(
                            0,
                            None
                            if x not in out_dynamic_dict.keys()
                            else dynamic_info.retrieve_output_dim(
                                input_shapes, j, i, x
                            ),
                        )
                        for i, x in enumerate(output_array.shape)
                    )
                ]
                for j, (output_array, out_dynamic_dict) in enumerate(
                    zip(output_arrays, dynamic_info.outputs)
                )
            )

        return output_arrays


class PytorchApacheTVMInferenceLearner(
    BaseArrayApacheTVMInferenceLearner, PytorchBaseInferenceLearner
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

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
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
        input_shapes = (
            [tuple(input_tensor.shape) for input_tensor in input_tensors]
            if self.network_parameters.dynamic_info is not None
            else None
        )
        output_arrays = self._inner_predict(input_arrays, input_shapes)
        return tuple(
            torch.from_numpy(array).to(self.device.to_torch_format())
            for array in output_arrays
        )

    @staticmethod
    def _convert_device(device: Any):
        if isinstance(device, int):
            return "cpu"
        return device


class TensorflowApacheTVMInferenceLearner(
    BaseArrayApacheTVMInferenceLearner, TensorflowBaseInferenceLearner
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
        input_arrays = (input_tensor.numpy() for input_tensor in input_tensors)
        input_shapes = (
            [tuple(input_tensor.shape) for input_tensor in input_tensors]
            if self.network_parameters.dynamic_info is not None
            else None
        )
        return tuple(
            tf.convert_to_tensor(out)
            for out in self._inner_predict(input_arrays, input_shapes)
        )


class NumpyApacheTVMInferenceLearner(
    BaseArrayApacheTVMInferenceLearner, NumpyBaseInferenceLearner
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

    def run(self, *input_tensors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[ndarray]): Input tensors belonging to the
                same batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[ndarray]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (input_tensor for input_tensor in input_tensors)
        input_shapes = (
            [tuple(input_tensor.shape) for input_tensor in input_tensors]
            if self.network_parameters.dynamic_info is not None
            else None
        )
        return tuple(self._inner_predict(input_arrays, input_shapes))


APACHE_TVM_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ApacheTVMInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchApacheTVMInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowApacheTVMInferenceLearner,
    DeepLearningFramework.NUMPY: NumpyApacheTVMInferenceLearner,
}
