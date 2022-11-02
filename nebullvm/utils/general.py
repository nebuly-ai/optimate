import sys
from packaging import version
from types import ModuleType

from nebullvm.base import DeepLearningFramework


def check_module_version(
    module: ModuleType, min_version: str = None, max_version: str = None
) -> bool:
    installed_version = module.__version__

    if min_version is not None:
        if version.parse(installed_version) < version.parse(min_version):
            return False

    if max_version is not None:
        if version.parse(installed_version) > version.parse(max_version):
            return False

    return True


def is_python_version_3_10():
    return (
        str(sys.version_info.major) + "." + str(sys.version_info.minor)
        == "3.10"
    )


def gpu_is_available(dl_framework: DeepLearningFramework):
    from nebullvm.utils.onnx import onnx_is_gpu_available
    from nebullvm.utils.tf import tensorflow_is_gpu_available
    from nebullvm.utils.torch import torch_is_gpu_available

    if dl_framework is DeepLearningFramework.PYTORCH:
        return torch_is_gpu_available()
    elif dl_framework is DeepLearningFramework.TENSORFLOW:
        return tensorflow_is_gpu_available()
    else:
        return onnx_is_gpu_available()
