from typing import Any, Union, Iterable, Sequence

from nebullvm.operations.base import Operation
from nebullvm.tools.base import ExecutionResult, Status


class FetchModelFromLocal(Operation):
    def execute(self, model: Any) -> ExecutionResult:
        self.state["model"] = model
        return ExecutionResult(Status.OK, None)

    def get_model(self) -> any:
        return self.state.get("model")

    def is_result_available(self) -> bool:
        pass


class FetchDataFromLocal(Operation):
    def execute(self, data: Union[Iterable, Sequence]) -> ExecutionResult:
        self.state["data"] = data
        return ExecutionResult(Status.OK, None)

    def get_data(self) -> any:
        return self.state.get("data")

    def is_result_available(self) -> bool:
        pass
