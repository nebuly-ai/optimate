import abc

from nebullvm.operations.base import Operation


class BuildInferenceLearner(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.inference_learner = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    def is_result_available(self) -> bool:
        pass
