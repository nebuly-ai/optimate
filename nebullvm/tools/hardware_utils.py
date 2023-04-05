import platform

import cpuinfo
import psutil

from nebullvm.core.models import HardwareSetup
from nebullvm.optional_modules.utils import torch_is_available, tensorflow_is_available
from nebullvm.tools.pytorch import torch_get_device_name
from nebullvm.tools.tf import tensorflow_get_gpu_name
from nebullvm.tools.utils import gpu_is_available


def get_hw_setup() -> HardwareSetup:
    gpu = None
    if gpu_is_available():
        gpu = get_gpu_name()
    return HardwareSetup(
        cpu=cpuinfo.get_cpu_info()["brand_raw"],
        operating_system=platform.system(),
        memory_gb=round(psutil.virtual_memory().total * 1e-9, 2),
        gpu=gpu,
    )


def get_gpu_name() -> str:
    if torch_is_available():
        name = torch_get_device_name()
    elif tensorflow_is_available():
        name = tensorflow_get_gpu_name()
    else:
        name = "Unknown"
    return name
