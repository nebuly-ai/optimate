import abc
import logging
from typing import Dict, Any


class Operation(abc.ABC):
    def __init__(self):
        self._state = {}
        self.execute_count = 0
        self.logger = logging.getLogger("nebullvm_logger")

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError()

    @property
    def state(self) -> Dict[str, any]:
        return self._state
