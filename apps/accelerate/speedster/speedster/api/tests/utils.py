import os
from pathlib import Path

from nebullvm.operations.conversions.pytorch import convert_torch_to_onnx
from nebullvm.tools.base import ModelParams, Device, DeviceType
from nebullvm.tools.data import DataManager
from nebullvm.tools.utils import gpu_is_available


def torch_to_onnx(model, input_data, output_path):
    model_params = ModelParams(1, [], [])
    output_path = os.path.join(output_path, "model.onnx")
    device = Device(DeviceType.GPU if gpu_is_available() else DeviceType.CPU)
    convert_torch_to_onnx(
        model, DataManager(input_data), model_params, Path(output_path), device
    )

    return output_path
