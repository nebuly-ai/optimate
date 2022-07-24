import cpuinfo
import torch

from nebullvm.base import ModelCompiler


def tvm_is_available() -> bool:
    try:
        import tvm  # noqa F401

        return True
    except ImportError:
        return False


def select_compilers_from_hardware_onnx():
    compilers = [ModelCompiler.ONNX_RUNTIME]
    if tvm_is_available():
        compilers.append(ModelCompiler.APACHE_TVM)
    if torch.cuda.is_available():
        compilers.append(ModelCompiler.TENSOR_RT)
    cpu_raw_info = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" in cpu_raw_info:
        compilers.append(ModelCompiler.OPENVINO)
    return compilers
