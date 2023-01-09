import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict

import numpy as np
from loguru import logger

from nebullvm.operations.optimizations.compressors.base import Compressor
from nebullvm.optional_modules.torch import torch, Module
from nebullvm.tools.data import DataManager
from nebullvm.tools.pytorch import save_with_torch_fx, load_with_torch_fx
from nebullvm.tools.venv import run_in_different_venv


def _save_model(model: Module, path: Path):
    try:
        save_with_torch_fx(model, path)
    except Exception as ex:
        logger.warning(
            f"Got an error while exporting with TorchFX. The model will be "
            f"saved using the standard PyTorch save pickling method. Error "
            f"got: {ex}"
        )
        torch.save(model, path / "model.pt")
        return path / "model.pt"
    else:
        return path


def _load_model(path: Path):
    if path.is_file():
        return torch.load(path)
    else:
        return load_with_torch_fx(path)


def _save_dataset(input_data: DataManager, path: Path):
    path.mkdir(exist_ok=True)
    for i, x in enumerate(input_data):
        torch.save(x, path / f"input_{i}.pt")


def _save_json(dictionary: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(dictionary, f)


def _write_requirements_file(path: Path):
    requirements = "sparseml\nsparsify\ntqdm"
    with open(path, "w") as f:
        f.write(requirements)


class SparseMLCompressor(Compressor):
    def execute(
        self,
        model: Module,
        train_input_data: DataManager,
        eval_input_data: DataManager,
        metric_drop_ths: float,
        metric: Callable,
    ):
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
                torch.cuda.is_available(),
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

            self.compressed_model = _load_model(pruned_model_path)

            if self.compressed_model is not None:
                error = self._compute_error(
                    model, self.compressed_model, eval_input_data, metric
                )
                if error > metric_drop_ths:
                    self.compressed_model = None
                else:
                    self.new_metric_ths = metric_drop_ths - error

    @staticmethod
    @torch.no_grad()
    def _compute_error(
        model: Module,
        pruned_model: Module,
        eval_input_data: DataManager,
        metric: Callable,
    ) -> float:
        if len(eval_input_data) == 0:
            return np.inf
        metric_val = 0.0
        model.eval()
        pruned_model.eval()
        for inputs, y in eval_input_data:
            if torch.cuda.is_available():
                inputs = tuple(data.cuda() for data in inputs)
                pruned_model.cuda()
                model.cuda()
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
