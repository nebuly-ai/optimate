import abc
import logging
from typing import Dict, Any, Optional, Union

from nebullvm.base import Device
from nebullvm.utils.general import gpu_is_available

logger = logging.getLogger("nebullvm_logger")


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
        self.logger = logging.getLogger("nebullvm_logger")

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
