import os
import platform

import cpuinfo
import psutil

from nebullvm.core.models import HardwareSetup, Device, DeviceType
from nebullvm.optional_modules.torch_xla import xm
from nebullvm.optional_modules.utils import (
    torch_is_available,
    tensorflow_is_available,
)
from nebullvm.tools.pytorch import torch_get_device_name
from nebullvm.tools.tf import tensorflow_get_gpu_name
from nebullvm.tools.utils import (
    gpu_is_available,
    tpu_is_available,
    neuron_is_available,
)


def get_hw_setup(device: Device = None) -> HardwareSetup:
    accelerator = None
    if (
        device is not None and device.type is DeviceType.GPU
    ) or gpu_is_available():
        accelerator = _get_gpu_name()
    elif (
        device is not None and device.type is DeviceType.TPU
    ) or tpu_is_available():
        accelerator = _get_tpu_device_name()
    elif (
        device is not None and device.type is DeviceType.NEURON
    ) or neuron_is_available():
        accelerator = _get_neuron_device_name()
    return HardwareSetup(
        cpu=cpuinfo.get_cpu_info()["brand_raw"],
        operating_system=platform.system(),
        memory_gb=round(psutil.virtual_memory().total * 1e-9, 2),
        accelerator=accelerator,
    )


def _get_gpu_name() -> str:
    if torch_is_available():
        name = torch_get_device_name()
    elif tensorflow_is_available():
        name = tensorflow_get_gpu_name()
    else:
        name = "Unknown"
    return name


def _get_neuron_device_name() -> str:
    output = os.popen("lshw -businfo").read()
    neuron_name = "Unknown Neuron"
    for line in output.splitlines():
        if "neuron" in line.lower():
            words = line.split(" ")
            if len(words) > 2:
                neuron_name = " ".join(words[-2:])
                break
    return neuron_name


def _get_tpu_device_name() -> str:
    return xm.xla_device_hw(xm.xla_device())
