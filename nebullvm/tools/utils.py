import subprocess
import sys
from pathlib import Path

import numpy as np
from packaging import version
from types import ModuleType
from typing import (
    Tuple,
    Any,
    List,
    Dict,
    Union,
    Iterable,
    Sequence,
    Optional,
    Callable,
)

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import DeepLearningFramework, Device, ModelParams
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import (
    extract_info_from_np_data,
    get_output_sizes_onnx,
)
from nebullvm.tools.pytorch import (
    extract_info_from_torch_data,
    get_outputs_sizes_torch,
)
from nebullvm.tools.tf import (
    extract_info_from_tf_data,
    get_outputs_sizes_tf,
)


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


def get_dl_framework(model: Any):
    if isinstance(model, torch.nn.Module):
        return DeepLearningFramework.PYTORCH
    elif isinstance(model, tf.Module) and model is not None:
        return DeepLearningFramework.TENSORFLOW
    elif isinstance(model, str):
        if Path(model).is_file():
            return DeepLearningFramework.NUMPY
        else:
            raise FileNotFoundError(
                f"No file '{model}' found, please provide a valid path to "
                f"a model."
            )
    else:
        raise TypeError(f"Model type {type(model)} not supported.")


def check_input_data(input_data: Union[Iterable, Sequence]):
    try:
        assert len(input_data) > 0
        assert isinstance(input_data[0], tuple)
        assert isinstance(input_data[0][0], tuple)
        assert isinstance(
            input_data[0][0][0], (np.ndarray, torch.Tensor, tf.Tensor)
        )
        assert isinstance(
            input_data[0][1],
            (np.ndarray, torch.Tensor, tf.Tensor, int, float, type(None)),
        )
    except:  # noqa E722
        return False
    else:
        return True


def is_data_subscriptable(input_data: Union[Iterable, Sequence]):
    try:
        input_data[0]
    except:  # noqa E722
        return False
    else:
        return True


def extract_info_from_data(
    model: Any,
    input_data: DataManager,
    dl_framework: DeepLearningFramework,
    dynamic_info: Optional[Dict],
    device: Device,
):
    batch_size, input_sizes, input_types, dynamic_info = INFO_EXTRACTION_DICT[
        dl_framework
    ](
        model,
        input_data,
        batch_size=None,
        input_sizes=None,
        input_types=None,
        dynamic_axis=dynamic_info,
        device=device,
    )
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=[
            {"size": size, "dtype": dtype}
            for size, dtype in zip(input_sizes, input_types)
        ],
        output_sizes=OUTPUT_SIZE_COMPUTATION_DICT[dl_framework](
            model, input_data[0][0], device
        ),
        dynamic_info=dynamic_info,
    )
    return model_params


def is_huggingface_data(data_sample: Any) -> bool:
    if is_dict_type(data_sample):
        return True
    elif isinstance(data_sample, str):
        return True
    elif isinstance(data_sample[0], str):
        return True
    return False


def is_dict_type(data_sample: Any):
    try:
        data_sample.items()
    except AttributeError:
        return False
    else:
        return True


INFO_EXTRACTION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: extract_info_from_torch_data,
    DeepLearningFramework.TENSORFLOW: extract_info_from_tf_data,
    DeepLearningFramework.NUMPY: extract_info_from_np_data,
}

OUTPUT_SIZE_COMPUTATION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: get_outputs_sizes_torch,
    DeepLearningFramework.TENSORFLOW: get_outputs_sizes_tf,
    DeepLearningFramework.NUMPY: get_output_sizes_onnx,
}
