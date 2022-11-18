from typing import Any

from nebullvm.operations.base import Operation


class Quantizer(Operation):
    def __init__(self):
        super().__init__()
        self.quantized_model = None

    def execute(self, **kwargs):
        raise NotImplementedError()

    def get_result(self) -> Any:
        return self.quantized_model
