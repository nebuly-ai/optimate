import abc


class App(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()
