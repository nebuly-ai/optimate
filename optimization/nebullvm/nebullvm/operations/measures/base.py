import abc

from nebullvm.operations.base import Operation


class Measure(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.measure_result = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()
