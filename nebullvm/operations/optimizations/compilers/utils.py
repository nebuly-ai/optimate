from nebullvm.tools.base import ModelCompiler, Device, DeviceType


def onnxruntime_is_available() -> bool:
    try:
        import onnxruntime  # noqa F401

        return True
    except ImportError:
        return False


def tvm_is_available() -> bool:
    try:
        import tvm  # noqa F401

        return True
    except ImportError:
        return False


def bladedisc_is_available() -> bool:
    try:
        import torch_blade  # noqa F401

        return True
    except ImportError:
        return False


def tensorrt_is_available() -> bool:
    try:
        import tensorrt  # noqa F401
        import polygraphy  # noqa F401

        return True
    except ImportError:
        return False


def torch_tensorrt_is_available() -> bool:
    try:
        import torch_tensorrt  # noqa F401

        return True
    except ImportError:
        return False


def openvino_is_available() -> bool:
    try:
        from openvino.runtime import Core  # noqa F401
    except ImportError:
        return False
    else:
        return True


def deepsparse_is_available() -> bool:
    try:
        import deepsparse  # noqa F401
    except ImportError:
        return False
    else:
        return True


def intel_neural_compressor_is_available() -> bool:
    try:
        import neural_compressor  # noqa F401
    except ImportError:
        return False
    else:
        return True


def select_compilers_from_hardware_onnx(device: Device):
    from nebullvm.optional_modules.utils import onnx_is_available

    compilers = []
    if onnx_is_available():
        if onnxruntime_is_available():
            compilers.append(ModelCompiler.ONNX_RUNTIME)
        if tvm_is_available():
            compilers.append(ModelCompiler.APACHE_TVM)
        if device.type is DeviceType.GPU and tensorrt_is_available():
            compilers.append(ModelCompiler.TENSOR_RT)
        if device.type is DeviceType.CPU and openvino_is_available():
            compilers.append(ModelCompiler.OPENVINO)
    return compilers


def select_compilers_from_hardware_torch(device: Device):
    from nebullvm.optional_modules.utils import torch_is_available

    compilers = []
    if torch_is_available():
        compilers.append(ModelCompiler.TORCHSCRIPT)
        if tvm_is_available():
            compilers.append(ModelCompiler.APACHE_TVM)
        if bladedisc_is_available():
            compilers.append(ModelCompiler.BLADEDISC)

        if device.type is DeviceType.CPU:
            if deepsparse_is_available():
                compilers.append(ModelCompiler.DEEPSPARSE)
            if intel_neural_compressor_is_available():
                compilers.append(ModelCompiler.INTEL_NEURAL_COMPRESSOR)
        elif device.type is DeviceType.GPU:
            if torch_tensorrt_is_available:
                compilers.append(ModelCompiler.TENSOR_RT)
    return compilers


def select_compilers_from_hardware_tensorflow():
    from nebullvm.optional_modules.utils import tensorflow_is_available

    compilers = []
    if tensorflow_is_available():
        compilers.append(ModelCompiler.XLA)
        compilers.append(ModelCompiler.TFLITE)
    return compilers
