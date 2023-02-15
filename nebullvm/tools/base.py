import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Union, Dict


class QuantizationType(Enum):
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"
    HALF = "HALF"


class Status(Enum):
    OK = "OK"
    ERROR = "ERROR"


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"


class DataType(str, Enum):
    FLOAT32 = "float32"
    INT32 = "int32"
    INT64 = "int64"


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


class ModelCompressor(Enum):
    SPARSE_ML = "sparseml"
    INTEL_PRUNING = "intel_pruning"


class OptimizationTime(Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"


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
    dynamic_info: Union[DynamicAxisInfo, Dict] = None

    def __post_init__(self):
        if isinstance(self.dynamic_info, dict):
            self.dynamic_info = DynamicAxisInfo(**self.dynamic_info)
        self.input_infos = [
            InputInfo(**x) if isinstance(x, dict) else x
            for x in self.input_infos
        ]

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
        if string == "cpu":
            return cls(DeviceType.CPU)
        elif string.startswith("cuda") or string.startswith("gpu"):
            return cls(
                DeviceType.GPU,
                int(string.split(":")[1] if ":" in string else 0),
            )
        else:
            raise Exception("Invalid device string")

    def to_torch_format(self) -> str:
        if self.type is DeviceType.CPU:
            return "cpu"
        return f"cuda:{self.idx}"

    def to_tf_format(self) -> str:
        if self.type is DeviceType.CPU:
            return "CPU"
        return f"GPU:{self.idx}"

    def get_total_memory(self) -> int:
        # Return total memory in bytes using nvidia-smi in bytes
        if self.type is DeviceType.CPU:
            raise Exception("CPU does not have memory")
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
        if self.type is DeviceType.CPU:
            raise Exception("CPU does not have memory")
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
