from pathlib import Path
from typing import Any

import torch
from nebullvm.operations.base import Operation

from open_alpha_tensor.config import BASE_CHECKPOINT_DATA_DIR, BASE_CHECKPOINT_DIR
from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
from open_alpha_tensor.core.training import Trainer


def optimizer_to(optim: torch.optim.Optimizer, device: str):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class LoadCheckPointOp(Operation):
    """An operation which loads a checkpoint during training of an
    OpenAlphaTensor model."""

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
        """Load a checkpoint from a directory.

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into.
            checkpoint_dir: The directory to load the checkpoint from.
        """
        checkpoint_dir = checkpoint_dir or BASE_CHECKPOINT_DIR
        if (
            Path(checkpoint_dir).exists()
            and len(list(Path(checkpoint_dir).glob("*.pt"))) > 0
        ):

            def key_func(x):
                return int(x.stem.split("_")[-1])

            checkpoint_path = sorted(
                Path(checkpoint_dir).glob("*.pt"), key=key_func
            )[-1]
            print(f"Loading checkpoint from {checkpoint_path}")
            old_device = model.device
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(old_device)
            print(f"Loaded model to {old_device}")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_to(optimizer, old_device)
            last_epoch = int(checkpoint_path.stem.split("_")[-1])
        else:
            last_epoch = 0

        self._last_epoch = last_epoch
        self._model = model
        self._optimizer = optimizer

    def get_last_epoch(self) -> int:
        """Returns the last epoch of the loaded checkpoint."""
        return self._last_epoch

    def get_model(self) -> AlphaTensorModel:
        """Returns the model loaded from the checkpoint."""
        return self._model

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Returns the optimizer loaded from the checkpoint."""
        return self._optimizer

    def get_result(self) -> Any:
        pass


class LoadCheckpointDataOp(Operation):
    """An operation which loads the games played while training an
    OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._loaded = False

    def execute(self, games_store_dir: Path, trainer: Trainer):
        """Load the games played while training an OpenAlphaTensor model.

        Args:
            games_store_dir: The directory where the games are stored.
            trainer: The trainer to load the games into.
        """
        games_store_dir = games_store_dir or BASE_CHECKPOINT_DATA_DIR
        # if games_store_dir contains games, load them
        if (
            games_store_dir.exists()
            and (games_store_dir / "game_data.json").exists()
        ):
            trainer.dataset.load_games(games_store_dir)
        self._loaded = True

    def get_result(self) -> bool:
        """Returns whether the games were loaded or not."""
        return self._loaded
