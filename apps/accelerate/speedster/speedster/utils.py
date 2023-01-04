import os
import platform
import torch
import uuid
from pathlib import Path
from typing import Any, Union

import cpuinfo
import psutil

from nebullvm.operations.optimizations.utils import load_model
from nebullvm.optional_modules.torch import Module
from nebullvm.optional_modules.utils import (
    torch_is_available,
    tensorflow_is_available,
)
from nebullvm.tools.base import Device
from nebullvm.tools.pytorch import torch_get_device_name
from nebullvm.tools.tf import tensorflow_get_gpu_name


def generate_model_id(model_name: str):
    return f"{str(uuid.uuid4())}_{hash(model_name)}"


def get_model_name(model: Any):
    if isinstance(model, str) or isinstance(model, Path):
        return str(model)
    else:
        return model.__class__.__name__


def _get_gpu_name():
    if torch_is_available():
        name = torch_get_device_name()
    elif tensorflow_is_available():
        name = tensorflow_get_gpu_name()
    else:
        name = "Unknown GPU"

    return name


def get_hw_info(device: Device):
    hw_info = {
        "cpu": cpuinfo.get_cpu_info()["brand_raw"],
        "operative_system": platform.system(),
        "ram": f"{round(psutil.virtual_memory().total * 1e-9, 2)} GB",
    }
    if device is Device.GPU:
        hw_info["gpu"] = _get_gpu_name()
    return hw_info


def read_model_size(model: Any):
    if isinstance(model, str) or isinstance(model, Path):
        size = os.stat(str(model)).st_size
    elif isinstance(model, Module):
        size = sum(
            param.nelement() * param.element_size()
            for param in model.parameters()
        )
    else:
        # we assume it is a tf_model
        size = model.count_params() * 4  # assuming full precision 32 bit
    return f"{round(size * 1e-6, 2)} MB"

def save_yolov5_model(model, path: Union[Path, str]):
    model.save(path)

def load_yolov5_model(path: Union[Path, str]):
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    optimized_model = load_model(path)
    last_layer = list(yolo_model.model.model.model.children())[-1]
    yolo_optimized = _OptimizedYolo(optimized_model, last_layer)
    yolo_model.model.model = yolo_optimized
    return yolo_model

class _OptimizedYolo(torch.nn.Module):
    def __init__(self, optimized_core, head_layer):
        super().__init__()
        self.core = optimized_core
        self.head = head_layer
    
    def forward(self, x, *args, **kwargs):
        x = list(self.core(x)) # it's a tuple
        return self.head(x)