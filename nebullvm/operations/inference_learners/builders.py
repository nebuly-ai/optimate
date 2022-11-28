from pathlib import Path
from typing import Union

import torch

from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.operations.inference_learners.deepsparse import (
    PytorchDeepSparseInferenceLearner,
)
from nebullvm.operations.inference_learners.neural_compressor import (
    PytorchNeuralCompressorInferenceLearner,
)
from nebullvm.operations.inference_learners.onnx import ONNX_INFERENCE_LEARNERS
from nebullvm.operations.inference_learners.openvino import (
    NumpyOpenVinoInferenceLearner,
)
from nebullvm.operations.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
    PytorchNvidiaInferenceLearner,
)
from nebullvm.operations.inference_learners.tensorflow import (
    TensorflowBackendInferenceLearner,
    TFLiteBackendInferenceLearner,
)
from nebullvm.operations.inference_learners.tvm import (
    PytorchApacheTVMInferenceLearner,
    NumpyApacheTVMInferenceLearner,
)
from nebullvm.optional_modules.tensor_rt import tensorrt as trt
from nebullvm.optional_modules.torch import ScriptModule, Module, GraphModule
from nebullvm.optional_modules.tvm import tvm
from nebullvm.optional_modules.openvino import CompiledModel
from nebullvm.tools.base import DeepLearningFramework, ModelParams, Device
from nebullvm.tools.onnx import get_input_names, get_output_names
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    VerifyContiguity,
)


class PytorchBuildInferenceLearner(BuildInferenceLearner):
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


class PyTorchTensorRTBuildInferenceLearner(BuildInferenceLearner):
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


class ONNXTensorRTBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: Union[str, Path],
        model_orig: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        nvidia_logger = trt.Logger(trt.Logger.ERROR)
        input_names = get_input_names(str(model_orig))
        output_names = get_output_names(str(model_orig))

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


class IntelNeuralCompressorBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: GraphModule,
        model_orig: Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        self.inference_learner = PytorchNeuralCompressorInferenceLearner(
            model=model_orig,
            model_quant=model,
            input_tfms=input_tfms,
            network_parameters=model_params,
            device=self.device,
        )


class PyTorchApacheTVMBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: torch.nn.Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        target_device = (
            str(tvm.target.cuda()) if self.device is Device.GPU else "llvm"
        )
        dev = tvm.device(str(target_device), 0)

        input_names = [
            f"input_{i}" for i in range(len(model_params.input_infos))
        ]

        graph_executor_module = tvm.contrib.graph_executor.GraphModule(
            model["default"](dev)
        )
        self.inference_learner = PytorchApacheTVMInferenceLearner(
            input_tfms=input_tfms,
            network_parameters=model_params,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=model,
            target=target_device,
        )


class ONNXApacheTVMBuildInferenceLearner(BuildInferenceLearner):
    def execute(
        self,
        model: torch.nn.Module,
        model_orig: Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        **kwargs,
    ):
        target_device = (
            str(tvm.target.cuda()) if self.device is Device.GPU else "llvm"
        )
        dev = tvm.device(str(target_device), 0)

        input_names = get_input_names(str(model_orig))

        graph_executor_module = tvm.contrib.graph_executor.GraphModule(
            model["default"](dev)
        )
        self.inference_learner = NumpyApacheTVMInferenceLearner(
            input_tfms=input_tfms,
            network_parameters=model_params,
            graph_executor_module=graph_executor_module,
            input_names=input_names,
            lib=model,
            target=target_device,
        )
