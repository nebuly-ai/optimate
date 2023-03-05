import subprocess
import sys
from pathlib import Path
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

import numpy as np
from loguru import logger
from packaging import version

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import (
    DeepLearningFramework,
    Device,
    ModelParams,
    DeviceType,
)
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
    axis_list: List[Dict],
):
    for idx, (tensor, size) in enumerate(zip(tensors, sizes)):
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
        if len(input_data[0]) > 1:
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


def check_dynamic_info_inputs(
    dynamic_info: Optional[Dict], input_sample: Tuple[Any]
):
    if dynamic_info is not None:
        assert dynamic_info.get("inputs") is not None, (
            "Dynamic info must contain an 'inputs' key with a list of "
            "dictionaries as value."
        )

        num_dynamic_inputs = len(dynamic_info["inputs"])
        num_model_inputs = len(input_sample)
        assert len(dynamic_info["inputs"]) == len(input_sample), (
            f"The number of dynamic inputs provided in the dynamic info "
            f"dict ({num_dynamic_inputs}) is not equal to the number "
            f"of inputs of the model ({num_model_inputs}). Detected model "
            f"input shapes are: {[input.shape for input in input_sample]} "
        )

        assert dynamic_info.get("outputs") is not None, (
            "Dynamic info must contain an 'outputs' key with a list of "
            "dictionaries as value."
        )


def extract_info_from_data(
    model: Any,
    input_data: DataManager,
    dl_framework: DeepLearningFramework,
    dynamic_info: Optional[Dict],
    device: Device,
):
    check_dynamic_info_inputs(dynamic_info, input_data.get_list(1)[0])
    batch_size, input_sizes, input_types, dynamic_info = INFO_EXTRACTION_DICT[
        dl_framework
    ](
        model,
        input_data,
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


def check_device(device: Optional[str]) -> Device:
    if device is None:
        if gpu_is_available():
            device = Device(DeviceType.GPU)
        else:
            device = Device(DeviceType.CPU)
    else:
        if any(x in device.lower() for x in ["cuda", "gpu"]):
            device_info = device.split(":")
            if len(device_info) == 2 and device_info[1].isdigit():
                idx = int(device_info[1])
            else:
                idx = 0
            if not gpu_is_available():
                logger.warning(
                    "Selected GPU device but no available GPU found on this "
                    "platform. CPU will be used instead. Please make sure "
                    "that the gpu is installed and can be used by your "
                    "framework."
                )
                device = Device(DeviceType.CPU)
            else:
                device = Device(DeviceType.GPU, idx=idx)
        else:
            device = Device(DeviceType.CPU)

    return device


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
