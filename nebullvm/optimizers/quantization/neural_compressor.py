from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch.fx
import torch.nn

import yaml
from torch.utils.data import DataLoader

from nebullvm.base import QuantizationType
from nebullvm.compressors.intel import IPCDataset
from nebullvm.utils.data import DataManager

try:
    from neural_compressor.experimental import (
        Quantization,
    )
except ImportError:
    Quantization = object
except ValueError:
    # MacOS
    Quantization = object


def _prepare_quantization_config(model: Any, tmp_dir: str):
    config = {
        "model": {
            "name": model.__class__.__name__,
            "framework": "pytorch_fx",
        },
        "evaluation": {"accuracy": {"metric": {"topk": 1}}},
        "tuning": {
            "accuracy_criterion": {"relative": 0.01},
        },
    }

    path_file = Path(tmp_dir) / "temp_qt.yaml"
    with open(path_file, "w") as f:
        yaml.dump(config, f)

    return path_file


def _get_dataloader(input_data: DataManager):
    bs = input_data[0][0][0].shape[0]
    ds = IPCDataset(input_data)
    dl = DataLoader(ds, bs)
    return dl


def _quantize_static(
    model: torch.nn.Module, input_data: DataManager
) -> torch.fx.GraphModule:
    with TemporaryDirectory() as tmp_dir:
        config_file_qt = _prepare_quantization_config(model, tmp_dir)
        quantizer = Quantization(str(config_file_qt))
        quantizer.model = model
        quantizer.calib_dataloader = _get_dataloader(input_data)
        quantizer.eval_dataloader = _get_dataloader(input_data)
        compressed_model = quantizer()

    return compressed_model


def quantize_neural_compressor(
    model: torch.nn.Module,
    quantization_type: QuantizationType,
    input_data: DataManager,
) -> torch.fx.GraphModule:
    if quantization_type is QuantizationType.STATIC:
        compressed_model = _quantize_static(model, input_data)
    else:
        raise NotImplementedError()

    return compressed_model
