import abc
from functools import partial
from typing import Dict, Any, List, Optional, Iterator

from nebullvm.core import util
from nebullvm.core.hosts import Host
from nebullvm.core.queues import ErrorQueue, ExecuteQueue, DeltaQueue
from nebullvm.utils.logger import get_logger

log = get_logger()


def _update_op_name(op: "Operation", name: str, parent: "Operation"):
    op.name = f"{parent.name}.{name}"


class Operation(abc.ABC):
    def __init__(self, block: bool = False, host: Optional[Host] = None):
        self.__name = ""
        self.__error_queue: Optional[ErrorQueue] = None
        self.__execute_queue: Optional[ExecuteQueue] = None
        self.__delta_queue: Optional[DeltaQueue] = None
        self._state_vars = set()
        self._children_operations = set()
        self._setattr_fun = self._default_setattr
        self._on_stop_callbacks: List[callable] = []
        self.block = block
        self.host = host

    @staticmethod
    def __is_state_var(key, value) -> bool:
        def is_proxy() -> bool:
            from nebullvm.core import proxies
            if isinstance(value, proxies.DeltaSetattrProxy):
                return True
            if isinstance(value, proxies.AsyncExecuteProxy):
                return True
            if isinstance(value, partial):
                return value.func.__name__ == "_async_execute"
            return False

        if key.startswith('_'):
            return False
        if is_proxy():
            return False
        if isinstance(value, Host) or key == "host":
            return False

        return util.is_json_serializable(value)

    def _default_setattr(self, key, value):
        if isinstance(value, Operation):
            self._children_operations.add(key)
            _update_op_name(value, key, self)
        if self.__is_state_var(key, value):
            self._state_vars.add(key)
        super().__setattr__(key, value)

    def __setattr__(self, key, value):
        fun = getattr(self, "_setattr_fun", self._default_setattr)
        fun(key, value)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "vars": {var: getattr(self, var) for var in self._state_vars},
            "operations": {op: getattr(self, op).state for op in self._children_operations}
        }

    @property
    def name(self) -> str:
        return self.__name or "root"

    @name.setter
    def name(self, value: str):
        self.__name = value

    def set_state(self, state: Dict[str, Any]):
        for var, value in state["vars"].items():
            setattr(self, var, value)
        for op, op_state in state["operations"].items():
            getattr(self, op).set_state(op_state)

    @property
    def children_operations(self) -> List["Operation"]:
        return [getattr(self, k) for k in self._children_operations]

    def stop(self):
        # Stop children
        for op in self.children_operations:
            op.stop()
        # Run callbacks and stop workers
        for c in self._on_stop_callbacks:
            c()

    def on_stop(self, callback: callable):
        self._on_stop_callbacks.append(callback)

    def visit_tree(self) -> Iterator["Operation"]:
        yield self
        for op in self.children_operations:
            yield from op.visit_tree()

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @property
    def error_queue(self) -> ErrorQueue:
        if self.__error_queue is None:
            self.__error_queue = self.host.queue_factory.new_error_queue()
        return self.__error_queue

    @property
    def delta_queue(self) -> DeltaQueue:
        if self.__delta_queue is None:
            self.__delta_queue = self.host.queue_factory.new_delta_queue()
        return self.__delta_queue

    @property
    def execute_queue(self) -> ExecuteQueue:
        if self.__execute_queue is None:
            self.__execute_queue = self.host.queue_factory.new_execute_queue()
        return self.__execute_queue
