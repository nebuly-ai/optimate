from pathlib import Path
from typing import Union
from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.inference_learners.deepsparse import (
    PytorchDeepSparseInferenceLearner,
)
from nebullvm.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.inference_learners.openvino import NumpyOpenVinoInferenceLearner
from nebullvm.inference_learners.tensor_rt import (
    PytorchNvidiaInferenceLearner,
    PytorchTensorRTInferenceLearner,
)
from nebullvm.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.optional_modules.torch import ScriptModule
from nebullvm.optional_modules.tensor_rt import tensorrt as trt
from nebullvm.optional_modules.openvino import CompiledModel
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.transformations.tensor_tfms import VerifyContiguity
from nebullvm.utils.onnx import get_input_names, get_output_names


class PytorchBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

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


class TensorflowBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(
        self,
        model,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TensorflowBackendInferenceLearner(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class TFLiteBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(
        self,
        model,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = TFLiteBackendInferenceLearner(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            device=self.device,
        )


class DeepSparseBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        **kwargs,
    ):
        input_names = get_input_names(str(model))
        output_names = get_output_names(str(model))

        self.inference_learner = PytorchDeepSparseInferenceLearner(
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
        )


class ONNXBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        dl_framework: DeepLearningFramework,
        **kwargs,
    ):
        input_names = get_input_names(str(model))
        output_names = get_output_names(str(model))

        self.inference_learner = ONNX_INFERENCE_LEARNERS[dl_framework](
            onnx_path=model,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            input_tfms=input_tfms,
            device=self.device,
        )


class OpenVINOBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(
        self,
        model: CompiledModel,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        infer_request = model.create_infer_request()

        input_keys = list(map(lambda obj: obj.get_any_name(), model.inputs))
        output_keys = list(map(lambda obj: obj.get_any_name(), model.outputs))

        self.inference_learner = NumpyOpenVinoInferenceLearner(
            compiled_model=model,
            infer_request=infer_request,
            input_keys=input_keys,
            output_keys=output_keys,
            input_tfms=input_tfms,
            network_parameters=model_params,
        )


class TensorRTBuildInferenceLearner(BuildInferenceLearner):
    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework

    def execute(self, *args, **kwargs):
        if self.dl_framework is DeepLearningFramework.PYTORCH:
            build_op = PyTorchTensorRTBuildInferenceLearner()
        elif self.dl_framework is DeepLearningFramework.NUMPY:
            build_op = ONNXTensorRTBuildInferenceLearner()
        else:
            raise ValueError(
                f"TensorRT is not supported for {self.dl_framework} models."
            )

        build_op.to(self.device).execute(*args, **kwargs)
        self.inference_learner = build_op.inference_learner


class PyTorchTensorRTBuildInferenceLearner(TensorRTBuildInferenceLearner):
    def __init__(self):
        super().__init__(DeepLearningFramework.PYTORCH)

    def execute(
        self,
        model: Union[str, Path],
        input_tfms: MultiStageTransformation,
        model_params: ModelParams,
        **kwargs,
    ):
        self.inference_learner = PytorchTensorRTInferenceLearner(
            torch_model=model,
            input_tfms=input_tfms,
            network_parameters=model_params,
            device=self.device,
        )


class ONNXTensorRTBuildInferenceLearner(TensorRTBuildInferenceLearner):
    def __init__(self):
        super().__init__(DeepLearningFramework.PYTORCH)

    def execute(
        self,
        model: Union[str, Path],
        onnx_model: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        nvidia_logger = trt.Logger(trt.Logger.ERROR)
        input_names = get_input_names(str(onnx_model))
        output_names = get_output_names(str(onnx_model))

        input_tfms.append(VerifyContiguity())
        runtime = trt.Runtime(nvidia_logger)
        engine = runtime.deserialize_cuda_engine(model)

        self.inference_learner = PytorchNvidiaInferenceLearner(
            engine=engine,
            input_tfms=input_tfms,
            network_parameters=model_params,
            input_names=input_names,
            output_names=output_names,
            nvidia_logger=nvidia_logger,
            device=self.device,
        )
