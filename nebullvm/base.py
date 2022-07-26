from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Union


class DataType(str, Enum):
    FLOAT = "float"
    INT = "int"


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
                if value == output_tag:
                    return input_shape[key]
        return default_output_value


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


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"


class ModelCompiler(Enum):
    TENSOR_RT = "tensor RT"
    OPENVINO = "openvino"
    APACHE_TVM = "tvm"
    ONNX_RUNTIME = "onnxruntime"
    DEEPSPARSE = "deepsparse"
    TORCHSCRIPT = "torchscript"
    TFLITE = "tflite"
    BLADEDISC = "bladedisc"


class QuantizationType(Enum):
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"
    QAT = "QAT"
    HALF = "HALF"


class OptimizationTime(Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"
