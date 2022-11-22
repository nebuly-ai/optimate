from pathlib import Path
from typing import List, Union
from nebullvm.base import ModelParams
from nebullvm.inference_learners import PytorchONNXInferenceLearner
from nebullvm.inference_learners.deepsparse import PytorchDeepSparseInferenceLearner
from nebullvm.inference_learners.openvino import NumpyOpenVinoInferenceLearner
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.optional_modules.torch import ScriptModule
from nebullvm.optional_modules.openvino import Model, CompiledModel
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.onnx import get_input_names, get_output_names


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
        **kwargs,

    ):
        input_names=get_input_names(str(model))
        output_names=get_output_names(str(model))

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
        input_tfms: MultiStageTransformation,
        **kwargs,

    ):  
        input_names=get_input_names(str(model))
        output_names=get_output_names(str(model))

        self.inference_learner = PytorchONNXInferenceLearner(
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            input_tfms=input_tfms,
            device=self.device,
        )

class OpenVINOBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: CompiledModel,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,

    ):
        infer_request = model.create_infer_request()

        input_keys = list(
            map(lambda obj: obj.get_any_name(), model.inputs)
        )
        output_keys = list(
            map(lambda obj: obj.get_any_name(), model.outputs)
        )
        
        self.inference_learner = NumpyOpenVinoInferenceLearner(
            compiled_model=model,
            infer_request=infer_request,
            input_keys=input_keys,
            output_keys=output_keys,
            input_tfms=input_tfms,
            network_parameters=model_params,
        )
