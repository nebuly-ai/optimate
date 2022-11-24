import subprocess
import sys

from packaging import version
from types import ModuleType
from typing import Tuple, Any, List, Dict


def ifnone(target, new_value):
    if target is None:
        return new_value
    else:
        return target


def inspect_dynamic_size(
    tensors: Tuple[Any, ...],
    sizes: List[Tuple[int, ...]],
    batch_size: int,
    axis_list: List[Dict],
):
    for idx, (tensor, size) in enumerate(zip(tensors, sizes)):
        size = (batch_size, *size)
        for idy, (j, k) in enumerate(zip(tensor.shape, size)):
            if j != k:
                if idy == 0:
                    tag = "batch_size"
                else:
                    tag = f"val_{j}_{k}"
                axis_list[idx][idy] = tag


def gpu_is_available():
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False


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
