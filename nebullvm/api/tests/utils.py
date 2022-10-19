import os

from nebullvm.base import ModelParams
from nebullvm.converters.torch_converters import convert_torch_to_onnx
from nebullvm.utils.data import DataManager


def torch_to_onnx(model, input_data, output_path):
    model_params = ModelParams(1, [], [])
    output_path = os.path.join(output_path, "model.onnx")
    convert_torch_to_onnx(
        model, model_params, output_path, DataManager(input_data)
    )

    return output_path
