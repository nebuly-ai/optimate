from typing import Any

from nebullvm.operations.base import Operation
from torch.utils.data import DataLoader


class FetchTrainingDataFromLocal(Operation):
    def get_result(self) -> Any:
        pass

    def execute(self, train_data: DataLoader, test_data: DataLoader):
        self.state["train_data"] = train_data
        self.state["test_data"] = test_data

    def get_train_data(self) -> DataLoader:
        return self.state.get("train_data")

    def get_test_data(self) -> DataLoader:
        return self.state.get("test_data")
