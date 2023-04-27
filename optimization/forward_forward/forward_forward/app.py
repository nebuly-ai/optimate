from nebullvm.apps.base import App

from forward_forward.root_op import ForwardForwardRootOp


class ForwardForwardApp(App):
    def __init__(self):
        super().__init__()
        self.root_op = ForwardForwardRootOp()

    def execute(self, *args, **kwargs):
        return self.root_op.execute(*args, **kwargs)
