import abc
from typing import List

from nebullvm.operations.base import Operation
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.tools.base import ExecutionResult


class Optimizer(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimized_models = None

    @abc.abstractmethod
    def execute(self, **kwargs) -> ExecutionResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_compilers(self) -> List[Compiler]:
        raise NotImplementedError()

    def is_result_available(self) -> bool:
        return self.optimized_models is not None
