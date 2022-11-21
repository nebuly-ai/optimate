from pathlib import Path
from typing import List, Union
from nebullvm.base import ModelParams
from nebullvm.inference_learners import ONNXInferenceLearner, PytorchONNXInferenceLearner
from nebullvm.inference_learners.deepsparse import PytorchDeepSparseInferenceLearner
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
        **kwargs,
    ):
        self.inference_learner = PytorchBackendInferenceLearner(
            torch_model=model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class DeepSparseBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        input_names: List[str],
        output_names: List[str],
        **kwargs,

    ):
        self.inference_learner = PytorchDeepSparseInferenceLearner(
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
        )


class ONNXBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        input_names: List[str],
        output_names: List[str],
        input_tfms: MultiStageTransformation,
        **kwargs,

    ):
        self.inference_learner = PytorchONNXInferenceLearner(
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            input_tfms=input_tfms,
            device=self.device,
        )
