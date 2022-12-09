import enum
import multiprocessing
from dataclasses import dataclass, field
from queue import Empty
from threading import Thread
from time import time
from typing import Any, Optional, Dict, Protocol, Tuple, Generic, Union, TypeVar, List

from deepdiff import Delta

from nebullvm.utils.logger import get_logger

log = get_logger()


class Queue(Protocol):
    def put(self, item: Any) -> None:
        pass

    def get(self, timeout_seconds: Optional[float] = None) -> Any:
        pass


class SingleProcessQueue:
    def put(self, item: Any) -> None:
        pass

    def get(self, timeout_seconds: Optional[float] = None) -> Any:
        pass


class HttpQueue:
    def put(self, item: Any) -> None:
        pass

    def get(self, timeout_seconds: Optional[float] = None) -> Any:
        pass


class MultiprocessingQueue:
    def __init__(self):
        self._queue = multiprocessing.Queue()

    def get(self, timeout_seconds: Optional[float] = None) -> Any:
        return self._queue.get(timeout=timeout_seconds)

    def put(self, item: Any) -> None:
        self._queue.put(item)


@dataclass
class ExecuteRequest:
    args: Tuple[any] = field(default_factory=tuple)
    kwargs: Dict[str, any] = field(default_factory=dict)
    state: Dict[str, any] = field(default_factory=dict)


@dataclass
class OperationDelta:
    op: str
    delta: Delta

    def __len__(self):
        return self.delta.to_dict().__len__()


class ExecuteQueue:
    def __init__(self, queue: Queue):
        self._queue = queue

    def get(self, timeout_seconds: Optional[float] = None) -> ExecuteRequest:
        return self._queue.get(timeout_seconds=timeout_seconds)

    def put(self, item: ExecuteRequest) -> None:
        self._queue.put(item)


class ErrorQueue:
    def __init__(self, queue: Queue):
        self._queue = queue

    def get(self, timeout_seconds: Optional[float] = None) -> Exception:
        return self._queue.get(timeout_seconds=timeout_seconds)

    def put(self, item: Exception) -> None:
        self._queue.put(item)


class DeltaQueue:
    def __init__(self, queue: Queue):
        self._queue = queue

    def get(self, timeout_seconds: Optional[float] = None) -> OperationDelta:
        return self._queue.get(timeout_seconds=timeout_seconds)

    def put(self, item: OperationDelta) -> None:
        self._queue.put(item)


class QueueKind(enum.Enum):
    MULTIPROCESSING = "multiprocessing"
    SINGLE_PROCESS = "single-process"
    HTTP = "http"


class QueueFactory:
    def __init__(self, kind: QueueKind):
        self.kind = kind

    def _new_queue(self) -> Queue:
        if self.kind is QueueKind.MULTIPROCESSING:
            return MultiprocessingQueue()
        if self.kind is QueueKind.HTTP:
            return HttpQueue()
        return SingleProcessQueue()

    def new_error_queue(self) -> ErrorQueue:
        return ErrorQueue(self._new_queue())

    def new_execute_queue(self) -> ExecuteQueue:
        return ExecuteQueue(self._new_queue())

    def new_delta_queue(self) -> DeltaQueue:
        return DeltaQueue(self._new_queue())


_T = TypeVar("_T", bound=Union[OperationDelta, ExecuteRequest, Exception])


class QueueCollector(Generic[_T]):
    """
    QueueCollector is a helper class that collects batches of items from a provided queue.
    """

    def __init__(self, source: Queue, collect_interval_seconds: float = 0.1):
        self.source = source
        self.collect_interval_seconds = collect_interval_seconds

    def collect(self) -> List[_T]:
        start = time()
        items = []
        while time() - start < self.collect_interval_seconds:
            try:
                delta = self.source.get(timeout_seconds=self.collect_interval_seconds)
                items.append(delta)
            except Empty:
                pass
        return items


class QueueSyncer(Thread):
    """
    QueueSyncer is a helper class that synchronizes two queues, by collecting items form a source and
    putting them into a sink.
    """

    def __init__(self, source: QueueCollector, sink: Queue):
        """
        Parameters
        ----------
        source: QueueCollector
            The collector that fetches items from the source queue.
        sink: Queue
            The queue to put the items into.
        """
        super().__init__(daemon=True)
        self.source = source
        self.sink = sink

    def run(self) -> None:
        while True:
            items = self.source.collect()
            for i in items:
                self.sink.put(i)

    def stop(self) -> None:
        log.debug("Stopping queue syncer")
        self.join(0)
        log.debug("Queue syncer stopped")
