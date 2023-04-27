from typing import Any, Union, Iterable, Sequence

from nebullvm.operations.base import Operation


class FetchModelFromLocal(Operation):
    def execute(self, model: Any):
        self.state["model"] = model

    def get_model(self) -> any:
        return self.state.get("model")

    def get_result(self) -> Any:
        pass


class FetchDataFromLocal(Operation):
    def execute(self, data: Union[Iterable, Sequence]):
        self.state["data"] = data

    def get_data(self) -> any:
        return self.state.get("data")

    def get_result(self) -> Any:
        pass
