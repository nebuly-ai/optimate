import abc
from typing import Dict, Any, Optional, Union

from loguru import logger

from nebullvm.tools.base import Device
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import gpu_is_available


def _check_device(device: Optional[str]) -> Device:
    if device is None:
        if gpu_is_available():
            device = Device.GPU
        else:
            device = Device.CPU
    else:
        if device.lower() == "gpu":
            if not gpu_is_available():
                logger.warning(
                    "Selected GPU device but no available GPU found on this "
                    "platform. CPU will be used instead. Please make sure "
                    "that the gpu is installed and can be used by your "
                    "framework."
                )
                device = Device.CPU
            else:
                device = Device.GPU
        else:
            device = Device.CPU

    return device


class Operation(abc.ABC):
    def __init__(self):
        self._state = {}
        self.device = Device.CPU
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
            self.device = _check_device(device)
        else:
            self.device = device
        return self
