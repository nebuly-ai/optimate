from typing import Optional

from nebullvm.core.models import ModelParams, Device
from nebullvm.operations.inference_learners.torchscript import (
    TorchScriptInferenceLearner,
)
from nebullvm.optional_modules.torch import ScriptModule
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class BladeDISCInferenceLearner(TorchScriptInferenceLearner):
    name = "BladeDISC"

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
