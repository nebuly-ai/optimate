from nebullvm.apps.base import App

from speedster.root_op import SpeedsterRootOp


class SpeedsterApp(App):
    def __init__(self):
        super().__init__()
        self.root_op = SpeedsterRootOp()

    def execute(self, *args, **kwargs):
        return self.root_op.execute(*args, **kwargs)
