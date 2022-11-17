from nebullvm.operations.base import Operation


class Quantizer(Operation):
    def __init__(self):
        super().__init__()
        self.quantized_model = None

    def execute(self, **kwargs):
        raise NotImplementedError()

    def is_result_available(self) -> bool:
        return self.quantized_model is not None
