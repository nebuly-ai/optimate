import os
import subprocess
import sys
import uuid
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

from nebullvm.core.models import (
    DeepLearningFramework,
    Device,
    ModelParams,
    DeviceType,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import (
    extract_info_from_np_data,
    get_output_info_onnx,
)
from nebullvm.tools.pytorch import (
    extract_info_from_torch_data,
    get_output_info_torch,
)
from nebullvm.tools.tf import (
    extract_info_from_tf_data,
    get_output_info_tf,
)


def get_model_size_mb(model: Any) -> float:
    if isinstance(model, str):
        size = os.stat(model).st_size
    elif isinstance(model, Path):
        size = os.path.getsize(model.as_posix())
    elif isinstance(model, torch.nn.Module):
        size = sum(p.nelement() * p.element_size() for p in model.parameters())
    else:
        # we assume it is a tf_model
        # assuming full precision 32 bit
        size = model.count_params() * 4
    return round(size * 1e-6, 2)


def get_model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, Path):
        return model.as_posix()
    return model.__class__.__name__


def generate_model_id(model: Any) -> str:
    model_name = get_model_name(model)
    return f"{str(uuid.uuid4())}_{hash(model_name)}"


def get_throughput(latency: float, batch_size: int = 1) -> float:
    if latency == 0:
        return -1
    return (1 / latency) * batch_size


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


def neuron_is_available():
    try:
        subprocess.check_output("neuron-ls")
        return True
    except Exception:
        return False


def tpu_is_available():
    # Check if a tpu is available
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        return xm.xla_device_hw(torch_xla.core.xla_model.xla_device()) == "TPU"
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
    is_diffusion: bool = False,
):
    check_dynamic_info_inputs(dynamic_info, input_data.get_list(1)[0])
    batch_size, input_sizes, input_types, dynamic_info = INFO_EXTRACTION_DICT[
        dl_framework
    ](
        model,
        input_data,
        dynamic_axis=dynamic_info,
        device=device,
        is_diffusion=is_diffusion,
    )

    output_infos = OUTPUT_INFO_COMPUTATION_DICT[dl_framework](
        model, input_data[0][0], device
    )
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=[
            {"size": size, "dtype": dtype}
            for size, dtype in zip(input_sizes, input_types)
        ],
        output_sizes=[info[0] for info in output_infos],
        output_types=[info[1] for info in output_infos],
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


def _get_idx(device: str) -> int:
    device_info = device.split(":")
    if len(device_info) == 2 and device_info[1].isdigit():
        idx = int(device_info[1])
    else:
        idx = 0
    return idx


def _set_device(
    accelerator_is_available: bool, device_type: DeviceType, idx: int
) -> Device:
    if not accelerator_is_available:
        logger.warning(
            f"Selected {device_type.name} device but no available "
            f"{device_type.name} found on this platform. CPU will "
            f"be used instead. Please make sure that the "
            f"{device_type.name} is installed and can be used by your "
            "framework."
        )
        device = Device(DeviceType.CPU)
    else:
        device = Device(device_type, idx=idx)

    return device


def check_device(device: Optional[str] = None) -> Device:
    if device is None:
        if gpu_is_available():
            device = Device(DeviceType.GPU)
        elif neuron_is_available():
            device = Device(DeviceType.NEURON)
        elif tpu_is_available():
            device = Device(DeviceType.TPU)
        else:
            device = Device(DeviceType.CPU)
    else:
        if any(x in device.lower() for x in ["cuda", "gpu"]):
            device = _set_device(
                accelerator_is_available=gpu_is_available(),
                device_type=DeviceType.GPU,
                idx=_get_idx(device),
            )
        elif "neuron" in device.lower():
            device = _set_device(
                accelerator_is_available=neuron_is_available(),
                device_type=DeviceType.NEURON,
                idx=_get_idx(device),
            )
        elif "tpu" in device.lower():
            device = _set_device(
                accelerator_is_available=tpu_is_available(),
                device_type=DeviceType.TPU,
                idx=_get_idx(device),
            )
        else:
            device = Device(DeviceType.CPU)

    return device


def get_gpu_compute_capability(gpu_idx: int) -> float:
    compute_capability = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    ).decode("utf-8")
    return float(compute_capability.split("\n")[gpu_idx])


INFO_EXTRACTION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: extract_info_from_torch_data,
    DeepLearningFramework.TENSORFLOW: extract_info_from_tf_data,
    DeepLearningFramework.NUMPY: extract_info_from_np_data,
}

OUTPUT_INFO_COMPUTATION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: get_output_info_torch,
    DeepLearningFramework.TENSORFLOW: get_output_info_tf,
    DeepLearningFramework.NUMPY: get_output_info_onnx,
}
