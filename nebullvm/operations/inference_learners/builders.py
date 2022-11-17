from nebullvm.base import Device, ModelParams
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.optional_modules.torch import ScriptModule


class PytorchBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        device: Device,
    ):
        self.inference_learner = PytorchBackendInferenceLearner(
            torch_model=model, network_parameters=model_params, device=device
        )
