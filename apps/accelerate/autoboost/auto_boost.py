from apps.accelerate.auto_boost.root_op import AutoBoostRootOp
from nebullvm.apps.base import App


class AutoBoostApp(App):
    def __init__(self):
        super().__init__()
        self.root_op = AutoBoostRootOp()

    def execute(self, *args, **kwargs):
        return self.root_op.to("GPU").execute(*args, **kwargs)
