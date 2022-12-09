import abc
import contextlib
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process
from threading import Thread
from typing import List, Type, Dict, Callable
from typing import Optional

from deepdiff import Delta

from nebullvm.core import util
from nebullvm.core.app import App, AppPhase
from nebullvm.core.hosts import RemoteHost, LocalHost
from nebullvm.core.operations import Operation
from nebullvm.core.proxies import AsyncExecuteProxy
from nebullvm.core.proxies import DeltaSetattrProxy
from nebullvm.core.queues import ExecuteQueue, ErrorQueue, DeltaQueue, OperationDelta, QueueCollector, QueueSyncer
from nebullvm.utils.logger import get_logger

log = get_logger()


@contextlib.contextmanager
def _state_deltas_generation(op: Operation, delta_queue: DeltaQueue):
    """Context manager for collecting operation state deltas and sending them to a delta queue."""
    proxy = DeltaSetattrProxy(op, delta_queue)
    with util.patched_attr(op, "_setattr_fun", proxy):
        yield


class _DeltaApplier(threading.Thread):
    """
    DeltaApplier is a daemon that collects state deltas from a source DeltaQueue and applies them to the operation
    state.

    Optionally, it can send any state update delta generated when modifying the operation state with the deltas
    received from the collector to an upstream DeltaQueue.
    """

    def __init__(self, op: Operation, collector: QueueCollector[OperationDelta], upstream: Optional[DeltaQueue] = None):
        """
        Parameters
        ----------
        op: Operation
            The operation to apply deltas to.
        collector: QueueCollector
            Collector that gives deltas to apply to the operation state.
        upstream: Optional[DeltaQueue]
            Optional delta queue to which send any state update delta generated when modifying the operation state
            with the deltas received from the collector.
        """
        super().__init__(daemon=True)
        self._op = op
        self._collector = collector
        self._upstream_queue = upstream

    def _get_op(self, name: str) -> Optional[Operation]:
        for op in self._op.visit_tree():
            if op.name == name:
                return op
        return None

    @staticmethod
    def _group_by_op(op_deltas: List[OperationDelta]) -> Dict[str, List[Delta]]:
        grouped = defaultdict(list)
        for op_delta in op_deltas:
            grouped[op_delta.op].append(op_delta.delta)
        return grouped

    def apply_op_deltas(self, op: Operation, deltas: List[Delta]):
        state = op.state
        for delta in deltas:
            state += delta
        if self._upstream_queue is not None:
            with _state_deltas_generation(op, self._upstream_queue):
                op.set_state(state)
        else:
            op.set_state(state)

    def _apply_deltas(self, deltas: List[OperationDelta]):
        log.debug(f"Applying {len(deltas)} deltas [op {self._op.name}]")
        for op_name, deltas in self._group_by_op(deltas).items():
            op = self._get_op(op_name)
            if op is None:
                log.error(f'Could not find operation "{op_name}" for delta [op {self._op.name}]')
                continue
            self.apply_op_deltas(op, deltas)

    def _run_once(self):
        deltas = self._collector.collect()
        if len(deltas) > 0:
            self._apply_deltas(deltas)

    def run(self, *args, **kwargs):
        log.debug(f"Starting delta applier [op {self._op.name}]")
        while True:
            self._run_once()

    def stop(self):
        self.join(0)


@dataclass
class AsyncRunner(abc.ABC):
    """Runner for executing operations asynchronously.

    The runner wraps the Operation execute method into an event loop that consumes an ExecuteQueue,
    and runs the "execute" function of the operation everytime there's a new execute request in the queue.

    If the Operation has children operation that need to be run asynchronously, the runner wraps their
    execute function during the setup of the event loop, so that they will be managed by an AsycnRunner too.
    """
    op: Operation
    execute_fn: callable
    execute_queue: ExecuteQueue
    delta_queue: DeltaQueue
    error_queue: ErrorQueue

    def _run_loop(self):
        self._setup()
        while True:
            try:
                self._run_once()
            except KeyboardInterrupt:
                pass
            except Exception as e:
                log.exception(e)
                self.error_queue.put(e)

    def _setup(self):
        for child in self.op.children_operations:
            wrap_execute_fn(
                op=child,
                execute_fn=child.execute,
                upstream_delta_queue=self.delta_queue,
                upstream_error_queue=self.error_queue,
            )

    def _run_once(self):
        # Wait for next execute request
        req = self.execute_queue.get()
        log.debug(f"Got execute request [op {self.op.name}]: {req}")

        # Temporarily replace __setattr__ method so that we can capture the state deltas
        # and send them to the delta queue
        with _state_deltas_generation(self.op, self.delta_queue):
            # Inject state
            self.op.set_state(req.state)
            # Execute the operation
            self.execute_fn(*req.args, **req.kwargs)

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass


class MultiThreadRunner(AsyncRunner):
    _thread: Optional[Thread] = field(default=None, init=False)

    def start(self):
        self._thread = Thread(target=self._run_loop, name=self.op.name, daemon=False)
        log.debug(f'Starting runner thread "{self._thread.name}" [op {self.op.name}]')
        self._thread.start()

    def stop(self):
        log.debug(f'Stopping runner thread "{self._thread.name}" [op {self.op.name}]')
        self._thread.join(0)
        log.debug(f'Runner thread stopped "{self._thread.name}" [op {self.op.name}]')


class MultiProcessRunner(AsyncRunner):
    _process: Optional[Process] = field(default=None, init=False)

    def start(self):
        self._process = Process(target=self._run_loop, name=self.op.name, daemon=False)
        log.debug(f'Starting runner process "{self._process.name}" [op {self.op.name}]')
        self._process.start()

    def stop(self):
        log.debug(f'Stopping runner process "{self._process.name}", [op {self.op.name}]')
        self._process.terminate()
        log.debug(f'Runner process stopped "{self._process.name}" [op {self.op.name}]')


def _async_execute(
        *args,
        op: Operation,
        execute_fn: Callable,
        upstream_delta_queue: Optional[DeltaQueue] = None,
        upstream_error_queue: Optional[ErrorQueue] = None,
        **kwargs,
):
    # Restore original execute function
    op.execute = execute_fn
    # If we are already in a sub-process,
    # then sync the error queue with the upstream one
    if upstream_error_queue is not None:
        error_collector = QueueCollector(op.error_queue)
        error_syncer = QueueSyncer(error_collector, upstream_error_queue)
        error_syncer.start()
    # Start async runner
    runner_cls: Type[AsyncRunner]
    if isinstance(op.host, RemoteHost):
        runner_cls = MultiThreadRunner
    else:
        runner_cls = MultiProcessRunner
    runner = runner_cls(
        op,
        execute_fn=execute_fn,
        execute_queue=op.execute_queue,
        delta_queue=op.delta_queue,
        error_queue=op.error_queue,
    )
    runner.start()
    op.on_stop(runner.stop)
    # Start DeltaApplier for syncing state deltas
    collector = QueueCollector(op.delta_queue)
    applier = _DeltaApplier(op=op, collector=collector, upstream=upstream_delta_queue)
    applier.start()
    op.on_stop(applier.stop)
    # Replace execute function with async proxy
    op.execute = AsyncExecuteProxy(op, op.execute_queue)
    # Run execute
    op.execute(*args, **kwargs)


def _wrap_children_execute(
        *args,
        op: Operation,
        execute_fn: callable,
        upstream_delta_queue: Optional[DeltaQueue] = None,
        upstream_error_queue: Optional[ErrorQueue] = None,
        **kwargs,
):
    for child in op.children_operations:
        child_execute_fn = child.execute
        if child.block is False:
            child.execute = partial(
                _async_execute,
                op=child,
                execute_fn=child_execute_fn,
                upstream_delta_queue=upstream_delta_queue,
                upstream_error_queue=upstream_error_queue,
            )
        else:
            child.execute = partial(
                _wrap_children_execute,
                op=child,
                execute_fn=child_execute_fn,
                upstream_delta_queue=upstream_delta_queue,
                upstream_error_queue=upstream_error_queue,
            )
    op.execute = execute_fn
    op.execute(*args, **kwargs)


def wrap_execute_fn(
        *,
        op: Operation,
        execute_fn: callable,
        upstream_delta_queue: Optional[DeltaQueue] = None,
        upstream_error_queue: Optional[ErrorQueue] = None,
):
    """Wrap the execute function of the provided operation and its children to support async execution.

    In case of async execution, the wrapping of the execute function of the operation children is done at
    runtime when the execute function of the parent operation is called, so that the resulting process gets managed
    by the process of the parent operation.
    """
    if op.block is False:
        new_execute_fn = partial(
            _async_execute,
            op=op,
            execute_fn=execute_fn,
            upstream_delta_queue=upstream_delta_queue,
            upstream_error_queue=upstream_error_queue,
        )
    else:
        new_execute_fn = partial(
            _wrap_children_execute,
            op=op,
            execute_fn=execute_fn,
            upstream_delta_queue=upstream_delta_queue,
            upstream_error_queue=upstream_error_queue,
        )
    op.execute = new_execute_fn


class AppRuntime:
    def __init__(self, app: App):
        self.app = app
        self.default_backend = LocalHost()

    def _attach_default_backend(self):
        for op in self.app.visit_tree():
            if op.host is None:
                op.host = self.default_backend

    def _setup(self):
        self._attach_default_backend()
        wrap_execute_fn(op=self.app.root_op, execute_fn=self.app.root_op.execute)

    def run(self):
        try:
            self._setup()
            self.app.run()
        except KeyboardInterrupt:
            self.app.stop()
        if self.app.phase is AppPhase.ERROR:
            sys.exit(1)
