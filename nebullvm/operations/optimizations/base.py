import abc
from typing import List

from nebullvm.operations.base import Operation
from nebullvm.operations.optimizations.compilers.base import Compiler


class Optimizer(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimized_models = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_compilers(self) -> List[Compiler]:
        raise NotImplementedError()

    def get_result(self) -> List:
        return self.optimized_models
