import copy
import re
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, Any, Callable, Optional, Tuple

import numpy as np
import torch.nn
import yaml
from torch.utils.data import DataLoader, Dataset

from nebullvm.compressors.base import BaseCompressor
from nebullvm.utils.data import DataManager
from nebullvm.utils.optional_modules import tensorflow as tf

try:
    from neural_compressor.experimental import Pruning
except ImportError:
    Pruning = object
except ValueError:
    # MacOS
    Pruning = object


def _get_model_framework(model: Any) -> str:
    if isinstance(model, torch.nn.Module):
        return "torch"
    elif isinstance(model, tf.Module) and model is not None:
        return "tensorflow"
    else:
        return "numpy"


class IntelPruningCompressor(BaseCompressor, ABC):
    def __init__(self, config_file: str = None):
        super().__init__(config_file)
        self._temp_dir = mkdtemp()

    @property
    def config_key(self) -> str:
        return "intel_pruning"

    @staticmethod
    def _get_default_config() -> Dict:
        # see https://github.com/intel/neural-compressor/blob/master/neural_compressor/conf/config.py  # noqa
        # for further details
        config = {
            "train": {
                "optimizer": {
                    "SGD": {"learning_rate": 0.001},
                },
                "criterion": {
                    "CrossEntropyLoss": {
                        "reduction": "mean",
                        "from_logits": False,
                    },
                },
                "epoch": 10,
                "start_epoch": 0,
                "end_epoch": 10,
                "iteration": 30,
                "execution_mode": "eager",  # either eager or graph
                # "hostfile": None,  # str for multinode training support
            },
            "approach": {
                "weight_compression": {
                    "initial_sparsity": 0.0,
                    "target_sparsity": 0.60,
                    "start_epoch": 0,
                    "end_epoch": 8,
                    "pruners": [
                        {
                            "start_epoch": 0,
                            "end_epoch": 8,
                            "prune_type": "basic_magnitude",
                        },
                    ],
                }
            },
        }
        return config

    def _prepare_pruning_config(self, model: Any):
        pruning_config = copy.deepcopy(self._config)
        framework = _get_model_framework(model)
        config = {
            "model": {
                "name": model.__class__.__name__,
                "framework": framework if framework != "torch" else "pytorch",
            },
            "evaluation": {"accuracy": {"metric": {"topk": 1}}},
            "device": "cpu",
            "tuning": {
                "random_seed": 1978,
                "tensorboard": False,
                "workspace": {"path": self._temp_dir},
            },
            "pruning": pruning_config,
        }
        path_file = Path(self._temp_dir) / "temp.yaml"
        with open(path_file, "w") as f:
            yaml.dump(config, f)
        with open(path_file, "r+") as f:
            file_str = f.read()
            file_str = re.sub(
                "pruners:\n      - end_epoch:",
                "pruners:\n      - !Pruner\n        end_epoch:",
                file_str,
            )
            f.seek(0)
            f.write(file_str)
        return path_file

    def compress(
        self,
        model: Any,
        train_input_data: DataManager,
        eval_input_data: DataManager,
        metric_drop_ths: float,
        metric: Callable,
    ) -> Tuple[Any, Optional[float]]:
        config_file_pr = self._prepare_pruning_config(model)
        prune = Pruning(str(config_file_pr))
        prune.model = model
        prune.train_dataloader = self._get_dataloader(train_input_data)
        prune.eval_dataloader = self._get_dataloader(eval_input_data)
        compressed_model = prune.fit()

        if compressed_model is None:
            return compressed_model, None
        error = self._compute_error(
            model, compressed_model, eval_input_data, metric
        )
        if error > metric_drop_ths:
            return None, None
        perf_loss_ths = metric_drop_ths - error
        return compressed_model, perf_loss_ths

    @abstractmethod
    def _compute_error(
        self,
        model: Any,
        compressed_model: Any,
        eval_input_data: DataManager,
        metric: Callable,
    ):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_dataloader(input_data: DataManager):
        raise NotImplementedError


class IPCDataset(Dataset):
    def __init__(self, input_data: DataManager):
        self.data = input_data
        self.batch_size = input_data[0][0][0].shape[0]

    def __len__(self):
        return sum([batch_inputs[0].shape[0] for batch_inputs, _ in self.data])

    def __getitem__(self, idx):
        batch_idx = int(idx / self.batch_size)
        item_idx = idx % self.batch_size
        data = tuple([data[item_idx] for data in self.data[batch_idx][0]])
        return data[0], self.data[batch_idx][1][item_idx]


class TorchIntelPruningCompressor(IntelPruningCompressor):
    @staticmethod
    def _get_dataloader(input_data: DataManager):
        bs = input_data[0][0][0].shape[0]
        ds = IPCDataset(input_data)
        dl = DataLoader(ds, bs)
        return dl

    def _compute_error(
        self,
        model: torch.nn.Module,
        compressed_model: torch.nn.Module,
        eval_input_data: DataManager,
        metric: Callable,
    ):
        if len(eval_input_data) == 0:
            return np.inf
        metric_val = 0
        for inputs, y in eval_input_data:
            pred_model = model(*inputs)
            pred_compressed_model = compressed_model(*inputs)
            metric_val += metric(pred_model, pred_compressed_model, y)
        return metric_val / len(eval_input_data)
