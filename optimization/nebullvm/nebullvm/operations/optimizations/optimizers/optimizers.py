import platform

from nebullvm.core.models import (
    DeepLearningFramework,
    DeviceType,
    ModelCompiler,
)
from nebullvm.operations.optimizations.optimizers.base import Optimizer
from nebullvm.operations.optimizations.compilers.utils import (
    tvm_is_available,
    bladedisc_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
    torch_tensorrt_is_available,
    onnxruntime_is_available,
    tensorrt_is_available,
    openvino_is_available,
    torch_neuron_is_available,
    torch_xla_is_available,
    faster_transformer_is_available,
)
from nebullvm.optional_modules.torch import torch
from nebullvm.optional_modules.utils import (
    torch_is_available,
    tensorflow_is_available,
    onnx_is_available,
)
from nebullvm.tools.utils import check_module_version


class PytorchOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.pipeline_dl_framework = DeepLearningFramework.PYTORCH

    def _select_compilers_from_hardware(self):
        compilers = []
        if torch_is_available():
            if self.device.type is DeviceType.TPU:
                if torch_xla_is_available():
                    compilers.append(ModelCompiler.TORCH_XLA)
                else:
                    raise RuntimeError(
                        "Torch XLA is not available on your platform. "
                        "Please install torch-xla the readme at this "
                        "link: https://github.com/pytorch/xla"
                    )
            elif self.device.type is DeviceType.NEURON:
                if torch_neuron_is_available():
                    compilers.append(ModelCompiler.TORCH_NEURON)
                else:
                    raise RuntimeError(
                        "Torch Neuron is not available on your platform. "
                        "Please install torch-neuron by following "
                        "this guide: https://awsdocs-neuron"
                        ".readthedocs-hosted.com/en/latest/general/"
                        "quick-start/torch-neuron.html."
                    )
            else:
                compilers.append(ModelCompiler.TORCHSCRIPT)
                if (
                    check_module_version(torch, min_version="2.0.0")
                    and platform.system() != "Windows"
                    and False
                ):  # Deactivated because save and load methods are
                    # not implemented
                    compilers.append(ModelCompiler.TORCH_DYNAMO)
                if tvm_is_available():
                    compilers.append(ModelCompiler.APACHE_TVM_TORCH)
                if bladedisc_is_available():
                    compilers.append(ModelCompiler.BLADEDISC)

                if self.device.type is DeviceType.CPU:
                    if deepsparse_is_available():
                        compilers.append(ModelCompiler.DEEPSPARSE)
                    if intel_neural_compressor_is_available():
                        compilers.append(ModelCompiler.INTEL_NEURAL_COMPRESSOR)
                elif self.device.type is DeviceType.GPU:
                    if torch_tensorrt_is_available():
                        compilers.append(ModelCompiler.TENSOR_RT_TORCH)
                    if faster_transformer_is_available():
                        compilers.append(ModelCompiler.FASTER_TRANSFORMER)
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
                compilers.append(ModelCompiler.APACHE_TVM_ONNX)
            if self.device.type is DeviceType.GPU and tensorrt_is_available():
                compilers.append(ModelCompiler.TENSOR_RT_ONNX)
            if self.device.type is DeviceType.CPU and openvino_is_available():
                compilers.append(ModelCompiler.OPENVINO)
        return compilers
