import abc
import logging
from typing import Dict

from nebullvm.tools.base import ExecutionResult


class Operation(abc.ABC):
    def __init__(self):
        self._state = {}
        self.execute_count = 0
        self.logger = logging.getLogger("nebullvm_logger")

    @abc.abstractmethod
    def execute(self, **kwargs) -> ExecutionResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_result_available(self) -> bool:
        raise NotImplementedError()

    @property
    def state(self) -> Dict[str, any]:
        return self._state
