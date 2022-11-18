from nebullvm.base import ModelParams
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.optional_modules.torch import ScriptModule
from nebullvm.transformations.base import MultiStageTransformation


class PytorchBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: ScriptModule,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
    ):
        self.inference_learner = PytorchBackendInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )
