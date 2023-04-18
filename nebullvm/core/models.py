import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional, Any, Union, Tuple, List, Dict

import numpy as np

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"


class QuantizationType(Enum):
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"
    HALF = "HALF"


class Status(Enum):
    OK = "OK"
    ERROR = "ERROR"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NEURON = "neuron"


class DataType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT32 = "int32"
    INT64 = "int64"

    @classmethod
    def from_framework_format(
        cls, dtype: Union[torch.dtype, tf.dtypes.DType, np.dtype]
    ):
        if isinstance(dtype, torch.dtype):
            framework = "torch"
        elif isinstance(dtype, tf.dtypes.DType):
            framework = "tensorflow"
        else:
            framework = "numpy"
            dtype = dtype.type
        return FRAMEWORK_TO_DATA_TYPE_CONVERSION_DICT[framework][dtype]

    def to_torch_format(self):
        for key, value in FRAMEWORK_TO_DATA_TYPE_CONVERSION_DICT[
            "torch"
        ].items():
            if value == self:
                return key

    def to_tf_format(self):
        for key, value in FRAMEWORK_TO_DATA_TYPE_CONVERSION_DICT[
            "tensorflow"
        ].items():
            if value == self:
                return key

    def to_numpy_format(self):
        for key, value in FRAMEWORK_TO_DATA_TYPE_CONVERSION_DICT[
            "numpy"
        ].items():
            if value == self:
                return key


class ModelCompiler(Enum):
    TENSOR_RT = "tensor_rt"
    TENSOR_RT_ONNX = "onnx_tensor_rt"
    TENSOR_RT_TORCH = "torch_tensor_rt"
    OPENVINO = "openvino"
    APACHE_TVM = "tvm"
    APACHE_TVM_TORCH = "torch_tvm"
    APACHE_TVM_ONNX = "onnx_tvm"
    ONNX_RUNTIME = "onnxruntime"
    DEEPSPARSE = "deepsparse"
    TORCHSCRIPT = "torchscript"
    XLA = "xla"
    TFLITE = "tflite"
    BLADEDISC = "bladedisc"
    INTEL_NEURAL_COMPRESSOR = "intel_neural_compressor"
    TORCH_NEURON = "torch_neuron"
    TORCH_XLA = "torch_xla"
    TORCH_DYNAMO = "torch_dynamo"
    FASTER_TRANSFORMER = "faster_transformer"


class ModelCompressor(Enum):
    SPARSE_ML = "sparseml"
    INTEL_PRUNING = "intel_pruning"


class OptimizationTime(Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"


@dataclass
class HardwareSetup:
    cpu: str
    operating_system: str
    memory_gb: int
    accelerator: Optional[str] = None


@dataclass
class OptimizedModel:
    inference_learner: Any
    latency_seconds: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    size_mb: float


@dataclass
class OriginalModel:
    model: Any
    latency_seconds: float
    throughput: float
    name: str
    size_mb: float
    framework: DeepLearningFramework


@dataclass
class BenchmarkOriginalModelResult:
    """The result of the LatencyOriginalModelMeasureOp"""

    latency_seconds: float
    model_outputs: Any


@dataclass
class OptimizeInferenceResult:
    """The result of the OptimizeInferenceOp"""

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_model: Optional[OptimizedModel]

    @property
    def metric_drop(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        return self.optimized_model.metric_drop

    @cached_property
    def latency_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.latency_seconds == 0:
            return -1
        return (
            self.original_model.latency_seconds
            / self.optimized_model.latency_seconds
        )

    @cached_property
    def throughput_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.original_model.throughput == 0:
            return -1
        return self.optimized_model.throughput / self.original_model.throughput

    @cached_property
    def size_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.size_mb == 0:
            return 1
        return self.original_model.size_mb / self.optimized_model.size_mb


class InputInfo:
    """Class for storing all the information needed for creating an input
    tensor for AI models.

    Attributes:
        size (tuple): Tuple with the input size (batch size excluded)
        dtype (str): Data type of the tensor.
        min_value (int or float, optional): Min value the tensor elements can
            have.
        max_value (int or float, optional): Max value the tensor elements can
            have.
    """

    def __init__(self, size: Tuple[int, ...], dtype: str, **extra_info):
        self.dtype = DataType(dtype)
        self.size = size
        self.__dict__.update(extra_info)

    def __getattr__(self, item):
        return self.__dict__.get(item)

    def dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


@dataclass
class DynamicAxisInfo:
    inputs: List[Dict[int, str]]
    outputs: List[Dict[int, str]]

    def dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    def retrieve_output_dim(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_idx: int,
        dimension_idx: int,
        default_output_value: int,
    ) -> int:
        output_tag = self.outputs[output_idx][dimension_idx]
        for input_dict, input_shape in zip(self.inputs, input_shapes):
            for key, value in input_dict.items():
                if (
                    isinstance(value, dict) and value.get("name") == output_tag
                ) or value == output_tag:
                    return input_shape[key]
        return default_output_value


@dataclass
class ModelParams:
    batch_size: int
    input_infos: List[InputInfo]
    output_sizes: List[Tuple[int, ...]]
    output_types: List[DataType]
    dynamic_info: Union[DynamicAxisInfo, Dict] = None

    def __post_init__(self):
        if isinstance(self.dynamic_info, dict):
            self.dynamic_info = DynamicAxisInfo(**self.dynamic_info)
        self.input_infos = [
            InputInfo(**x) if isinstance(x, dict) else x
            for x in self.input_infos
        ]
        self.output_types = [DataType(x) for x in self.output_types]

    def dict(self):
        def recursively_dictionarize(element):
            if isinstance(element, list):
                element = [recursively_dictionarize(el) for el in element]
            elif hasattr(element, "dict"):
                element = element.dict()
            return element

        return {
            k: recursively_dictionarize(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @property
    def input_sizes(self):
        for input_info in self.input_infos:
            yield input_info.size


class Device:
    def __init__(self, type: DeviceType, idx: int = 0):
        self.type = type
        self.idx = idx

    @classmethod
    def from_str(cls, string: str) -> "Device":
        if string.startswith("cuda") or string.startswith("gpu"):
            return cls(
                DeviceType.GPU,
                int(string.split(":")[1] if ":" in string else 0),
            )
        elif string.startswith("tpu"):
            return cls(
                DeviceType.TPU,
                int(string.split(":")[1] if ":" in string else 0),
            )

        return cls(DeviceType.CPU)

    def to_torch_format(self) -> str:
        if self.type is DeviceType.GPU:
            return f"cuda:{self.idx}"
        elif self.type is DeviceType.TPU:
            return f"xla:{self.idx}"

        return "cpu"

    def to_tf_format(self) -> str:
        if self.type is DeviceType.GPU:
            return f"GPU:{self.idx}"

        return "CPU"

    def get_total_memory(self) -> int:
        # Return total memory in bytes using nvidia-smi in bytes
        if self.type is not DeviceType.GPU:
            raise Exception("Device type must be GPU")
        else:
            try:
                output = (
                    subprocess.check_output(
                        "nvidia-smi --query-gpu=memory.total "
                        "--format=csv,nounits,noheader",
                        shell=True,
                    )
                    .decode("utf-8")
                    .split()[self.idx]
                )
                return int(output) * 1024 * 1024
            except Exception:
                raise Exception(
                    "Unable to get total memory of device. "
                    "Please make sure nvidia-smi is available."
                )

    def get_free_memory(self) -> int:
        # Return free memory in bytes using nvidia-smi in bytes
        if self.type is not DeviceType.GPU:
            raise Exception("Device type must be GPU")
        else:
            try:
                output = (
                    subprocess.check_output(
                        "nvidia-smi --query-gpu=memory.free "
                        "--format=csv,nounits,noheader",
                        shell=True,
                    )
                    .decode("utf-8")
                    .split()[self.idx]
                )
                return int(output) * 1024 * 1024
            except Exception:
                raise Exception(
                    "Unable to get free memory of device. "
                    "Please make sure nvidia-smi is available."
                )


FRAMEWORK_TO_DATA_TYPE_CONVERSION_DICT = {
    "torch": {
        torch.float16: DataType.FLOAT16,
        torch.float32: DataType.FLOAT32,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64,
    },
    "tensorflow": {
        tf.float16: DataType.FLOAT16,
        tf.float32: DataType.FLOAT32,
        tf.int32: DataType.INT32,
        tf.int64: DataType.INT64,
    },
    "numpy": {
        np.float16: DataType.FLOAT16,
        np.float32: DataType.FLOAT32,
        np.int32: DataType.INT32,
        np.int64: DataType.INT64,
    },
}
