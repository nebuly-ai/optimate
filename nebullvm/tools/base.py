from enum import Enum


class Status(Enum):
    OK = "OK"
    ERROR = "ERROR"


class DeepLearningFramework(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"
