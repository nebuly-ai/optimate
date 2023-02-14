import abc
from typing import Dict, Any, Union

from loguru import logger

from nebullvm.tools.base import Device, DeviceType
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import check_device


class Operation(abc.ABC):
    def __init__(self):
        self._state = {}
        self.device = Device(DeviceType.CPU)
        self.execute_count = 0
        self.logger = logger
        self.feedback_collector = None

    def set_feedback_collector(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        for value in self.__dict__.values():
            if isinstance(value, Operation):
                value.set_feedback_collector(feedback_collector)

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError()

    @property
    def state(self) -> Dict[str, any]:
        return self._state

    def to(self, device: Union[str, Device]):
        if isinstance(device, str):
            self.device = check_device(device)
        else:
            self.device = device
        return self
