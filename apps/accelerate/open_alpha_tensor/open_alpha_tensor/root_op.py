from nebullvm.operations.base import Operation


class OpenAlphaTensorRootOp(Operation):
    def __init__(self):
        super().__init__()

        self._build_model_op = None
        self._load_checkpoint_op = None
        self._training_op = None
