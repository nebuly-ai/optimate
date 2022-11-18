import abc
from typing import Tuple, Optional

from nebullvm.operations.base import Operation


class Measure(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.measure_result = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_result(self) -> Optional[Tuple]:
        raise NotImplementedError()
