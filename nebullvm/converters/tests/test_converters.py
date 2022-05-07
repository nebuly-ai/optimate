from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.base import ModelParams
from nebullvm.converters import ONNXConverter


class TorchTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=256, kernel_size=3
        )
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(256, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor):
        x = self.relu(self.conv(input_tensor))
        x = x.sum(dim=(-2, -1))
        return self.sigmoid(self.linear(x))


@pytest.mark.parametrize("ai_model", [TorchTestModel()])
def test_onnx_converter(ai_model):
    model_params = ModelParams(
        batch_size=1,
        input_infos=[{"size": (3, 256, 256), "dtype": "float"}],
        output_sizes=[(2,)],
    )
    converter = ONNXConverter(model_name="test_model")
    with TemporaryDirectory() as tmp_dir:
        converted_model_path = converter.convert(
            ai_model, model_params=model_params, save_path=Path(tmp_dir)
        )
        assert converted_model_path.exists()
