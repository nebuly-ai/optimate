from typing import Optional

from nebullvm.base import ModelParams, Device
from nebullvm.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)
from nebullvm.optional_modules.torch import ScriptModule
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


class BladeDISCInferenceLearner(PytorchBackendInferenceLearner):
    @classmethod
    def from_torch_model(
        cls,
        model: ScriptModule,
        network_parameters: ModelParams,
        device: Device,
        input_tfms: Optional[MultiStageTransformation] = None,
        input_data: DataManager = None,
    ):
        return cls(
            torch_model=model,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
            input_data=input_data,
            device=device,
        )
