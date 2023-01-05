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
