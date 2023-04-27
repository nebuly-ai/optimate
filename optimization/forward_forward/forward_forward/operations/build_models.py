from abc import ABC, abstractmethod

import torch

from nebullvm.operations.base import Operation

from forward_forward.utils.modules import (
    FCNetFFProgressive,
    RecurrentFCNetFF,
    LMFFNet,
)


class BaseModelBuildOperation(Operation, ABC):
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def execute(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        optimizer_params: dict,
        loss_fn_name: str,
        output_size: int = None,
    ):
        raise NotImplementedError

    def get_result(self):
        return self.model


class FCNetFFProgressiveBuildOperation(BaseModelBuildOperation):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        optimizer_params: dict,
        loss_fn_name: str,
        output_size: int = None,
    ):
        layer_sizes = [input_size] + [hidden_size] * n_layers
        model = FCNetFFProgressive(
            layer_sizes=layer_sizes,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_params,
            loss_fn_name=loss_fn_name,
            epochs=-1,
        )
        if output_size is not None:
            output_layer = torch.nn.Linear(layer_sizes[-1], output_size)
            model = torch.nn.Sequential(model, output_layer)

        self.model = model


class RecurrentFCNetFFBuildOperation(BaseModelBuildOperation):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        optimizer_params: dict,
        loss_fn_name: str,
        output_size: int = None,
    ):
        layer_sizes = [input_size] + [hidden_size] * n_layers + [output_size]
        model = RecurrentFCNetFF(
            layer_sizes=layer_sizes,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_params,
            loss_fn_name=loss_fn_name,
        )
        self.model = model


class LMFFNetBuildOperation(BaseModelBuildOperation):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        optimizer_params: dict,
        loss_fn_name: str,
        output_size: int = None,
    ):
        model = LMFFNet(
            token_num=output_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            seq_len=input_size,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_params,
            loss_fn_name=loss_fn_name,
            epochs=-1,
            predicted_tokens=-1,
        )
        self.model = model
