import copy
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, Any

import torch.nn
import tensorflow as tf
import yaml
from torch.utils.data import DataLoader, Dataset

from nebullvm.compressors.base import BaseCompressor
from nebullvm.utils.data import DataManager

try:
    from neural_compressor.experimental import Pruning
except ImportError:
    pass


def _get_model_framework(model: Any) -> str:
    if isinstance(model, torch.nn.Module):
        return "torch"
    elif isinstance(model, tf.Module):
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
                    "Adam": {
                        "learning_rate": 0.001,
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "epsilon": 1e-07,
                        "amsgrad": False,
                    },
                },
                "criterion": {
                    "SparseCategoricalCrossentropy": {
                        "reduction": "mean",
                        "from_logits": False,
                    },
                },
                # "dataloader": {
                #     "last_batch": "rollover",  # either rollover or discard
                #     "batch_size": 64,
                #     "dataset": {
                #         "dummy_v2": {
                #         },
                #     },
                #     "filter": {
                #
                #     },
                #     "transform": {
                #
                #     },
                #     "shuffle": False,
                #     "distributed": False,
                # },
                "epoch": 10,
                "start_epoch": 0,
                "end_epoch": 10,
                # "iteration": None,  # int
                # "frequency": None,  # int
                "execution_mode": "eager",  # either eager or graph
                # "postprocess": {
                #     "transform": {
                #
                #     },
                # },
                "hostfile": None,  # str for multinode training support
            },
            "approach": {
                "weight_compression": {
                    "initial_sparsity": 0.0,
                    "target_sparsity": 0.97,
                    "start_epoch": 0,
                    "end_epoch": 10,
                },
            },
        }
        return config

    def _prepare_config(self, model: Any):
        pruning_config = copy.deepcopy(self._config)
        config = {
            "model": {
                "name": model.__class__.name,
                "framework": _get_model_framework(model),
            },
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
        return path_file

    def compress(self, model: Any, input_data: DataManager, *args, **kwargs):
        config_file = self._prepare_config(model)
        prune = Pruning(config_file)
        prune.model = model
        prune.train_dataloader = self._get_dataloader(input_data)
        compressed_model = prune.fit()
        return compressed_model

    @staticmethod
    @abstractmethod
    def _get_dataloader(input_data: DataManager):
        raise NotImplementedError


class _IPCDataset(Dataset):
    def __init__(self, input_data: DataManager):
        self._input_data = input_data
        self._internal_size = input_data[0][0][0].shape[0]

    def __getitem__(self, item):
        ptr = item // self._internal_size
        return sum(self._input_data[ptr], ())

    def __len__(self):
        last_el_size = self._input_data[-1][0][0].shape[0]
        return self._internal_size * (len(self._input_data) - 1) + last_el_size


class TorchIntelPruningCompressor(IntelPruningCompressor):
    @staticmethod
    def _get_dataloader(input_data: DataManager):
        bs = input_data[0][0][0].shape[0]
        ds = _IPCDataset(input_data)
        dl = DataLoader(ds, bs)
        return dl
