import abc

from nebullvm.core.queues import QueueFactory, QueueKind


class Host(abc.ABC):
    def __init__(self, queue_factory: QueueFactory):
        self.queue_factory = queue_factory


class LocalHost(Host):
    def __init__(self):
        super().__init__(QueueFactory(kind=QueueKind.MULTIPROCESSING))


class RemoteHost(Host):
    def __init__(self):
        super().__init__(QueueFactory(kind=QueueKind.HTTP))
