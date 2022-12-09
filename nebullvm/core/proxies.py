from copy import deepcopy

from deepdiff import Delta, DeepDiff

from nebullvm.core.operations import Operation
from nebullvm.core.queues import ExecuteRequest, DeltaQueue, OperationDelta, ExecuteQueue


class AsyncExecuteProxy:
    """AsyncExecuteProxy is meant for replacing the execute method Operations to make it async.
    Instead of executing the operation directly, the proxy puts an execute request on the execute queue, which is
    consumed by a runner on a different thread/process.
    """

    def __init__(self, op: Operation, queue: ExecuteQueue):
        self.op = op
        self.queue = queue

    def __call__(self, *args, **kwargs):
        req = ExecuteRequest(
            args=args,
            kwargs=kwargs,
            state=self.op.state,
        )
        self.queue.put(req)


class DeltaSetattrProxy:
    """DeltaSetattrProxy replaces the __setattr__ method of the contained operation so that it can capture the state
    deltas and send them to the delta queue.
    """

    def __init__(self, op: Operation, delta_queue: DeltaQueue):
        self._delta_queue = delta_queue
        self._op = op

    def __call__(self, name: str, value: any):
        original_state = deepcopy(self._op.state)
        self._op._default_setattr(name, value) # noqa
        diff = DeepDiff(original_state, self._op.state)
        op_delta = OperationDelta(
            delta=Delta(diff),
            op=self._op.name,
        )
        # if delta is not empty then put into queue
        if len(op_delta) > 0:
            self._delta_queue.put(op_delta)
