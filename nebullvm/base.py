from dataclasses import dataclass
from enum import Enum
from typing import Tuple


@dataclass
class ModelParams:
    batch_size: int
    input_size: Tuple[int, ...]
    output_size: Tuple[int, ...]

    def dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"


class ModelCompiler(Enum):
    TENSOR_RT = "tensor RT"
    OPENVINO = "openvino"
    APACHE_TVM = "tvm"
