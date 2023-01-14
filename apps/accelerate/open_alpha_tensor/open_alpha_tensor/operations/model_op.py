import json
from pathlib import Path
from typing import Any

import torch
from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel


class BuildModelOp(Operation):
    """An operation which builds an OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._model = None

    def execute(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        """Builds the OpenAlphaTensor model.

        Args:
            tensor_length (int): Number of tensors to as history.
            input_size (int): Flattened size of the matrices to be multiplied.
            scalars_size (int): Size of the scalar vectors fed to the torso
            model.
            emb_dim (int): Embedding dimension.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            n_logits (int): Number of logits output by the policy head.
            n_samples (int): Number of samples used by the policy head at
            evaluation time.
        """
        self._model = AlphaTensorModel(
            tensor_length=tensor_length,
            input_size=input_size,
            scalars_size=scalars_size,
            emb_dim=emb_dim,
            n_steps=n_steps,
            n_logits=n_logits,
            n_samples=n_samples,
        )

    def get_model(self) -> AlphaTensorModel:
        """Returns the built model."""
        return self._model

    def get_result(self) -> Any:
        pass


class BuildOptimizerOp(Operation):
    """An operation which builds an optimizer for an OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._optimizer = None

    def execute(
        self,
        optimizer_name: str,
        model: AlphaTensorModel,
        lr: float,
        weight_decay: float,
    ):
        """Builds the optimizer for the OpenAlphaTensor model.

        Args:
            optimizer_name (str): Name of the optimizer used.
            model (AlphaTensorModel): OpenAlphaTensor model to be trained.
            lr (float): Learning rate.
            weight_decay (float): Weight decay used by the optimizer.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
        self._optimizer = optimizer

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Returns the built optimizer."""
        return self._optimizer

    def get_result(self) -> Any:
        pass


class SaveModelOp(Operation):
    """An operation which saves an OpenAlphaTensor model.
    The model parameters are stored in a json file, while the model weights
    are stored in a .pt file."""

    def get_result(self) -> Any:
        pass

    def execute(
        self,
        model: AlphaTensorModel,
        save_dir: str,
    ):
        """Saves the OpenAlphaTensor model.

        Args:
            model (AlphaTensorModel): OpenAlphaTensor model to be saved.
            save_dir (str): Directory where the model will be saved.
        """
        save_dir = Path(save_dir if save_dir else ".")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "final_model.pt")
        model_params = {
            "input_size": model.input_size,
            "tensor_length": model.tensor_length,
            "scalars_size": 1,
            "emb_dim": model.emb_dim,
            "n_steps": model.n_steps,
            "n_logits": model.n_logits,
            "n_samples": model.n_samples,
        }
        # save parameters in a json file
        with open(save_dir / "model_params.json", "w") as f:
            json.dump(model_params, f)
