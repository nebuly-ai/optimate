from dataclasses import dataclass
from enum import Enum
from typing import Any


class Status(Enum):
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class ExecutionResult:
    status: Status
    output: Any


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"
