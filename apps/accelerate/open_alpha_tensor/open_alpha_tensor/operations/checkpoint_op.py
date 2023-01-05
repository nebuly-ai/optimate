from pathlib import Path
from typing import Any

import torch
from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
from open_alpha_tensor.core.training import Trainer


class LoadCheckPointOp(Operation):
    def __init__(self):
        super().__init__()
        self._last_epoch = None
        self._model = None
        self._optimizer = None

    def execute(
        self,
        model: AlphaTensorModel,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
    ):
        if (
            checkpoint_dir is not None
            and Path(checkpoint_dir).exists()
            and len(list(Path(checkpoint_dir).glob("*.pt"))) > 0
        ):

            def key_func(x):
                return int(x.stem.split("_")[-1])

            checkpoint_path = sorted(
                Path(checkpoint_dir).glob("*.pt"), key=key_func
            )[-1]
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            last_epoch = int(checkpoint_path.stem.split("_")[-1])
        else:
            last_epoch = 0

        self._last_epoch = last_epoch
        self._model = model
        self._optimizer = optimizer

    def get_last_epoch(self) -> int:
        return self._last_epoch

    def get_model(self) -> AlphaTensorModel:
        return self._model

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def get_result(self) -> Any:
        pass


class LoadCheckpointDataOp(Operation):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def execute(self, games_store_dir: Path, trainer: Trainer):
        # if games_store_dir contains games, load them
        if (
            games_store_dir.exists()
            and (games_store_dir / "game_data.json").exists()
        ):
            trainer.dataset.load_games(games_store_dir)
        self._loaded = True

    def get_result(self) -> bool:
        return self._loaded
