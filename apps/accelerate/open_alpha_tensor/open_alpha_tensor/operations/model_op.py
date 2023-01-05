import json
from typing import Any

import torch
from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel


class BuildModelOp(Operation):
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
        return self._model

    def get_result(self) -> Any:
        pass


class BuildOptimizerOp(Operation):
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
        return self._optimizer

    def get_result(self) -> Any:
        pass


class SaveModelOp(Operation):
    def get_result(self) -> Any:
        pass

    def execute(
        self,
        model: AlphaTensorModel,
        matrix_size,
        action_memory: int,
        embed_dim: int,
        n_actions,
        actions_sampled: int,
    ):
        torch.save(model.state_dict(), "final_model.pt")
        model_params = {
            "input_size": matrix_size**2,
            "tensor_length": action_memory + 1,
            "scalars_size": 1,
            "emb_dim": embed_dim,
            "n_steps": 1,
            "n_logits": n_actions,
            "n_samples": actions_sampled,
        }
        # save parameters in a json file
        with open("model_params.json", "w") as f:
            json.dump(model_params, f)
