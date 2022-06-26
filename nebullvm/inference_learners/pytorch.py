from pathlib import Path
from typing import Tuple, Union, Optional

import torch

from nebullvm.base import ModelParams
from nebullvm.inference_learners import (
    PytorchBaseInferenceLearner,
    LearnerMetadata,
)
from nebullvm.transformations.base import MultiStageTransformation


class PytorchBackendInferenceLearner(PytorchBaseInferenceLearner):
    MODEL_NAME = "model_scripted.pt"

    def __init__(self, torch_model: torch.jit.ScriptModule, **kwargs):
        super().__init__(**kwargs)
        self.model = torch_model

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            res = self.model(*input_tensors)
            if not isinstance(res, tuple):
                return (res,)
            return res

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        self.model.save(path / self.MODEL_NAME)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        model = torch.jit.load(path / cls.MODEL_NAME)
        metadata = LearnerMetadata.read(path)
        return cls(
            torch_model=model,
            network_parameters=ModelParams(**metadata.network_parameters),
            input_tfms=metadata.input_tfms,
        )

    @classmethod
    def from_torch_model(
        cls,
        model: torch.nn.Module,
        network_parameters: ModelParams,
        input_tfms: Optional[MultiStageTransformation] = None,
    ):
        model_scripted = torch.jit.script(model, optimize=True)
        return cls(
            torch_model=model_scripted,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
        )
