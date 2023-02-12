from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.operations.optimizations.compilers.utils import (
    bladedisc_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
    onnxruntime_is_available,
    openvino_is_available,
    tensorrt_is_available,
    torch_tensorrt_is_available,
    tvm_is_available,
)
from nebullvm.optional_modules.utils import (
    faster_transformer_is_available,
    onnx_is_available,
    tensorflow_is_available,
    torch_is_available,
)
from nebullvm.tools.base import DeepLearningFramework, Device, ModelCompiler


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
            if faster_transformer_is_available():
                compilers.append(ModelCompiler.FASTER_TRANSFORMER)

            if self.device is Device.CPU:
                if deepsparse_is_available():
                    compilers.append(ModelCompiler.DEEPSPARSE)
                if intel_neural_compressor_is_available():
                    compilers.append(ModelCompiler.INTEL_NEURAL_COMPRESSOR)
            elif self.device is Device.GPU:
                if torch_tensorrt_is_available():
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
