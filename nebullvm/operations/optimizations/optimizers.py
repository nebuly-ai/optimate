from typing import Dict, Type

from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.operations.inference_learners.builders import (
    DeepSparseBuildInferenceLearner,
    OpenVINOBuildInferenceLearner,
    PytorchBuildInferenceLearner,
    ONNXBuildInferenceLearner,
    TensorflowBuildInferenceLearner,
    TFLiteBuildInferenceLearner,
    IntelNeuralCompressorBuildInferenceLearner,
    PyTorchTensorRTBuildInferenceLearner,
    ONNXTensorRTBuildInferenceLearner,
    PyTorchApacheTVMBuildInferenceLearner,
    ONNXApacheTVMBuildInferenceLearner,
)
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.deepsparse import (
    DeepSparseCompiler,
)
from nebullvm.operations.optimizations.compilers.intel_neural_compressor import (  # noqa: E501
    IntelNeuralCompressorCompiler,
)
from nebullvm.operations.optimizations.compilers.onnx import ONNXCompiler
from nebullvm.operations.optimizations.compilers.openvino import (
    OpenVINOCompiler,
)
from nebullvm.operations.optimizations.compilers.pytorch import (
    PytorchBackendCompiler,
)
from nebullvm.operations.optimizations.compilers.tensor_rt import (
    PyTorchTensorRTCompiler,
    ONNXTensorRTCompiler,
)
from nebullvm.operations.optimizations.compilers.tensorflow import (
    TensorflowBackendCompiler,
    TFLiteBackendCompiler,
)
from nebullvm.operations.optimizations.compilers.tvm import (
    PyTorchApacheTVMCompiler,
    ONNXApacheTVMCompiler,
)
from nebullvm.operations.optimizations.compilers.utils import (
    tvm_is_available,
    bladedisc_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
    torch_tensorrt_is_available,
    onnxruntime_is_available,
    tensorrt_is_available,
    openvino_is_available,
)
from nebullvm.optional_modules.utils import (
    torch_is_available,
    tensorflow_is_available,
    onnx_is_available,
)
from nebullvm.tools.base import DeepLearningFramework, ModelCompiler, Device


class PytorchOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.pipeline_dl_framework = DeepLearningFramework.PYTORCH

    def _select_compilers_from_hardware(self):
        compilers = []
        if torch_is_available():
            compilers.append(ModelCompiler.TORCHSCRIPT)
            if tvm_is_available():
                compilers.append(ModelCompiler.APACHE_TVM)
            if bladedisc_is_available():
                compilers.append(ModelCompiler.BLADEDISC)

            if self.device is Device.CPU:
                if deepsparse_is_available():
                    compilers.append(ModelCompiler.DEEPSPARSE)
                if intel_neural_compressor_is_available():
                    compilers.append(ModelCompiler.INTEL_NEURAL_COMPRESSOR)
            elif self.device is Device.GPU:
                if torch_tensorrt_is_available:
                    compilers.append(ModelCompiler.TENSOR_RT)
        return compilers


class TensorflowOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.pipeline_dl_framework = DeepLearningFramework.TENSORFLOW

    def _select_compilers_from_hardware(self):
        compilers = []
        if tensorflow_is_available():
            compilers.append(ModelCompiler.XLA)
            compilers.append(ModelCompiler.TFLITE)
        return compilers


class ONNXOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.pipeline_dl_framework = DeepLearningFramework.NUMPY

    def _select_compilers_from_hardware(self):
        compilers = []
        if onnx_is_available():
            if onnxruntime_is_available():
                compilers.append(ModelCompiler.ONNX_RUNTIME)
            if tvm_is_available():
                compilers.append(ModelCompiler.APACHE_TVM)
            if self.device is Device.GPU and tensorrt_is_available():
                compilers.append(ModelCompiler.TENSOR_RT)
            if self.device is Device.CPU and openvino_is_available():
                compilers.append(ModelCompiler.OPENVINO)
        return compilers


COMPILER_TO_OPTIMIZER_MAP: Dict[
    ModelCompiler, Dict[DeepLearningFramework, Type[Compiler]]
] = {
    ModelCompiler.TORCHSCRIPT: {
        DeepLearningFramework.PYTORCH: PytorchBackendCompiler
    },
    ModelCompiler.DEEPSPARSE: {
        DeepLearningFramework.PYTORCH: DeepSparseCompiler
    },
    ModelCompiler.INTEL_NEURAL_COMPRESSOR: {
        DeepLearningFramework.PYTORCH: IntelNeuralCompressorCompiler
    },
    ModelCompiler.TENSOR_RT: {
        DeepLearningFramework.PYTORCH: PyTorchTensorRTCompiler,
        DeepLearningFramework.NUMPY: ONNXTensorRTCompiler,
    },
    ModelCompiler.APACHE_TVM: {
        DeepLearningFramework.PYTORCH: PyTorchApacheTVMCompiler,
        DeepLearningFramework.NUMPY: ONNXApacheTVMCompiler,
    },
    ModelCompiler.ONNX_RUNTIME: {DeepLearningFramework.NUMPY: ONNXCompiler},
    ModelCompiler.OPENVINO: {DeepLearningFramework.NUMPY: OpenVINOCompiler},
    ModelCompiler.TFLITE: {
        DeepLearningFramework.TENSORFLOW: TFLiteBackendCompiler
    },
    ModelCompiler.XLA: {
        DeepLearningFramework.TENSORFLOW: TensorflowBackendCompiler
    },
}

COMPILER_TO_INFERENCE_LEARNER_MAP: Dict[
    ModelCompiler, Dict[DeepLearningFramework, Type[BuildInferenceLearner]]
] = {
    ModelCompiler.TORCHSCRIPT: {
        DeepLearningFramework.PYTORCH: PytorchBuildInferenceLearner
    },
    ModelCompiler.DEEPSPARSE: {
        DeepLearningFramework.PYTORCH: DeepSparseBuildInferenceLearner
    },
    ModelCompiler.INTEL_NEURAL_COMPRESSOR: {
        DeepLearningFramework.PYTORCH: IntelNeuralCompressorBuildInferenceLearner  # noqa: E501
    },
    ModelCompiler.TENSOR_RT: {
        DeepLearningFramework.PYTORCH: PyTorchTensorRTBuildInferenceLearner,
        DeepLearningFramework.NUMPY: ONNXTensorRTBuildInferenceLearner,
    },
    ModelCompiler.APACHE_TVM: {
        DeepLearningFramework.PYTORCH: PyTorchApacheTVMBuildInferenceLearner,
        DeepLearningFramework.NUMPY: ONNXApacheTVMBuildInferenceLearner,
    },
    ModelCompiler.ONNX_RUNTIME: {
        DeepLearningFramework.NUMPY: ONNXBuildInferenceLearner
    },
    ModelCompiler.OPENVINO: {
        DeepLearningFramework.NUMPY: OpenVINOBuildInferenceLearner
    },
    ModelCompiler.TFLITE: {
        DeepLearningFramework.TENSORFLOW: TFLiteBuildInferenceLearner
    },
    ModelCompiler.XLA: {
        DeepLearningFramework.TENSORFLOW: TensorflowBuildInferenceLearner
    },
}
