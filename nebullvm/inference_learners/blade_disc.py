from typing import Optional

from torch.jit import ScriptModule

from nebullvm.base import ModelParams
from nebullvm.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


class BladeDISCInferenceLearner(PytorchBackendInferenceLearner):
    @classmethod
    def from_torch_model(
        cls,
        model: ScriptModule,
        network_parameters: ModelParams,
        input_tfms: Optional[MultiStageTransformation] = None,
        input_data: DataManager = None,
    ):
        return cls(
            torch_model=model,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
            input_data=input_data,
        )
