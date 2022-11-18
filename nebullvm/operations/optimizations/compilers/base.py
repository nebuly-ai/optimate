import abc
from typing import Any

from nebullvm.operations.base import Operation


class Compiler(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.compiled_model = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compile_model(**kwargs) -> Any:
        raise NotImplementedError()

    def get_result(self) -> Any:
        return self.compiled_model
