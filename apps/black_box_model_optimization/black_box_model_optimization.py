from apps.black_box_model_optimization.root_op import (
    BlackBoxModelOptimizationRootOp,
)
from nebullvm.apps.base import App


class BlackBoxModelOptimization(App):
    def __init__(self):
        super().__init__()
        self.root_op = BlackBoxModelOptimizationRootOp()

    def execute(self, *args, **kwargs):
        return self.root_op.to("GPU").execute(*args, **kwargs)
