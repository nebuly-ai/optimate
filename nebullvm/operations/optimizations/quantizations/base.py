from nebullvm.operations.base import Operation
from nebullvm.tools.base import ExecutionResult


class Quantizer(Operation):
    def __init__(self):
        super().__init__()
        self.quantized_model = None

    def execute(self, **kwargs) -> ExecutionResult:
        raise NotImplementedError()

    def is_result_available(self) -> bool:
        return self.quantized_model is not None
