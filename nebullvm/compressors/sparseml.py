import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Tuple, Optional, Dict

import numpy as np
import torch
import torch.fx

from nebullvm.compressors.base import BaseCompressor
from nebullvm.utils.data import DataManager
from nebullvm.utils.venv import run_in_different_venv

FX_MODULE_NAME = "NebullvmFxModule"


def _save_with_torch_fx(model: torch.nn.Module, path: Path):
    traced_model = torch.fx.symbolic_trace(model)
    traced_model.to_folder(path, FX_MODULE_NAME)


def _load_with_torch_fx(path: Path):
    module_file = path / "module.py"
    with open(module_file, "r") as f:
        module_str = f.read()
    exec(module_str)
    model = eval(FX_MODULE_NAME)()
    model.load_state_dict(torch.load(path / "pruned_state_dict.pt"))
    return model


def _save_model(model: torch.nn.Module, path: Path):
    try:
        _save_with_torch_fx(model, path)
    except Exception:
        torch.save(model, path / "model.pt")
        return path / "model.pt"
    else:
        return path


def _load_model(path: Path):
    if path.is_file():
        return torch.load(path)
    else:
        return _load_with_torch_fx(path)


def _save_dataset(input_data: DataManager, path: Path):
    path.mkdir(exist_ok=True)
    for i, x in enumerate(input_data):
        torch.save(x, path / f"input_{i}.pt")


def _save_json(dictionary: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(dictionary, f)


def _write_requirements_file(path: Path):
    requirements = "torch<=1.9\ntorchvision<=0.10\nsparseml\nsparsify\ntqdm"
    with open(path, "w") as f:
        f.write(requirements)


class SparseMLCompressor(BaseCompressor):
    def compress(
        self,
        model: torch.nn.Module,
        train_input_data: DataManager,
        eval_input_data: DataManager,
        metric_drop_ths: float,
        metric: Callable,
    ) -> Tuple[Any, Optional[float]]:
        script_path = (
            Path(__file__).parent / "scripts/neural_magic_training.py"
        )
        with TemporaryDirectory(dir=".") as tmp_dir:
            tmp_dir = Path(tmp_dir)
            requirements_file = tmp_dir / "requirements.txt"
            model_path = _save_model(model, tmp_dir)
            training_data_dir = tmp_dir / "train"
            eval_data_dir = tmp_dir / "eval"
            config_file = tmp_dir / "config.json"
            pruned_model_path = (
                tmp_dir / "pruned_model.pt"
                if model_path.is_file()
                else tmp_dir
            )

            _write_requirements_file(requirements_file)
            _save_dataset(train_input_data, training_data_dir)
            _save_dataset(eval_input_data, eval_data_dir)
            _save_json(self._config, config_file)

            run_in_different_venv(
                str(requirements_file),
                str(script_path),
                "--model",
                f"{model_path}",
                "--train_dir",
                f"{training_data_dir}",
                "--eval_dir",
                f"{eval_data_dir}",
                "--config",
                f"{config_file}",
                "--pruned_model",
                f"{pruned_model_path}",
            )

            pruned_model = _load_model(pruned_model_path)

            error = self._compute_error(
                model, pruned_model, eval_input_data, metric
            )
            if error > metric_drop_ths:
                return None, None
            new_metric_ths = metric_drop_ths - error

        return pruned_model, new_metric_ths

    @staticmethod
    @torch.no_grad()
    def _compute_error(
        model: torch.nn.Module,
        pruned_model: torch.nn.Module,
        eval_input_data: DataManager,
        metric: Callable,
    ) -> float:
        if len(eval_input_data) == 0:
            return np.inf
        metric_val = 0.0
        model.eval()
        pruned_model.eval()
        for inputs, y in eval_input_data:
            model_pred = model(*inputs)
            pruned_pred = pruned_model(*inputs)
            metric_val += metric(model_pred, pruned_pred, y)
        return metric_val / len(eval_input_data)

    @staticmethod
    def _get_default_config() -> Dict:
        return {
            "training_epochs": 10,
            "epochs_pruning_window": {"start_epoch": 0, "end_epoch": 10},
            "loss_fn": "CrossEntropy",
            "lr": 1e-3,
            "momentum": 0.9,
        }

    @property
    def config_key(self) -> str:
        return "sparseml"
