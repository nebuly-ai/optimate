from nebullvm.base import ModelCompiler


def onnxruntime_is_available() -> bool:
    try:
        import onnxruntime  # noqa F401

        return True
    except ImportError:
        return False


def onnx_is_available() -> bool:
    try:
        import onnx  # noqa F401

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
        import openvino  # noqa F401
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


def select_compilers_from_hardware_onnx():
    compilers = []
    if onnx_is_available():
        if onnxruntime_is_available():
            compilers.append(ModelCompiler.ONNX_RUNTIME)
        if tvm_is_available():
            compilers.append(ModelCompiler.APACHE_TVM)
        if tensorrt_is_available():
            compilers.append(ModelCompiler.TENSOR_RT)
        if openvino_is_available():
            compilers.append(ModelCompiler.OPENVINO)
    return compilers
