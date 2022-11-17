import abc

from nebullvm.tools.base import ExecutionResult


class App(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def execute(self, **kwargs) -> ExecutionResult:
        raise NotImplementedError()
