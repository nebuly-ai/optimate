from nebullvm.apps.base import App
from nebullvm.operations.root.black_box_model_optimization import (
    BlackBoxModelOptimizationRootOp,
)


class BlackBoxModelOptimization(App):
    def __init__(self):
        super().__init__()
        self.root_op = BlackBoxModelOptimizationRootOp()

    def execute(self, *args, **kwargs):
        return self.root_op.execute(*args, **kwargs)
