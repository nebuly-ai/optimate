from typing import List, Tuple, Any, Optional, Dict

import numpy as np
from loguru import logger

from nebullvm.config import ONNX_PROVIDERS
from nebullvm.optional_modules.onnx import onnx
from nebullvm.optional_modules.onnxruntime import onnxruntime as ort
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import (
    InputInfo,
    DataType,
    DeepLearningFramework,
    Device,
    DeviceType,
)


def convert_to_numpy(tensor: Any):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
    elif isinstance(tensor, tf.Tensor) and tensor is not None:
        tensor = tensor.numpy()
    elif isinstance(tensor, int):
        tensor = np.array([tensor])
    else:
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(tensor)}")
    return tensor


def convert_to_target_framework(
    tensor: np.ndarray, framework: DeepLearningFramework
) -> Any:
    if framework is DeepLearningFramework.PYTORCH:
        return torch.from_numpy(tensor)
    elif framework is DeepLearningFramework.TENSORFLOW:
        return tf.convert_to_tensor(tensor)
    else:
        return tensor


def get_input_names(onnx_model: str):
    model = onnx.load(onnx_model)
    input_all = [node.name for node in model.graph.input]
    return input_all


def get_output_names(onnx_model: str):
    model = onnx.load(onnx_model)
    output_all = [node.name for node in model.graph.output]
    return output_all


def run_onnx_model(
    onnx_model: str, input_tensors: List[np.ndarray], device: Device
) -> List[np.ndarray]:
    from nebullvm.optional_modules.onnxruntime import onnxruntime as ort

    if device.type is DeviceType.GPU:
        ONNX_PROVIDERS["cuda"][1] = (
            "CUDAExecutionProvider",
            {
                "device_id": device.idx,
            },
        )

    model = ort.InferenceSession(
        onnx_model,
        providers=ONNX_PROVIDERS["cuda"][1:]
        if device.type is DeviceType.GPU
        else ONNX_PROVIDERS["cpu"],
    )
    inputs = {
        name: array
        for name, array in zip(get_input_names(onnx_model), input_tensors)
    }
    res = model.run(
        output_names=get_output_names(onnx_model), input_feed=inputs
    )
    return list(res)


def _extract_dynamic_axis(
    onnx_model: str,
    data: List[Tuple[Tuple[np.ndarray, ...], np.ndarray]],
    input_sizes: List[Tuple[int, ...]],
    device: Device,
    max_data: int = 100,
) -> Optional[Dict]:
    from nebullvm.tools.utils import inspect_dynamic_size

    dynamic_axis = {"inputs": [{}] * len(input_sizes), "outputs": []}
    output_sizes = []
    for i, input_data in enumerate(data):
        input_tensors = input_data[0]
        if i >= max_data:
            break
        inspect_dynamic_size(
            input_tensors, input_sizes, dynamic_axis["inputs"]
        )
        outputs = tuple(
            run_onnx_model(onnx_model, list(input_tensors), device)
        )
        if i == 0:
            dynamic_axis["outputs"] = [{}] * len(outputs)
            output_sizes = [tuple(output.shape[1:]) for output in outputs]
        inspect_dynamic_size(outputs, output_sizes, dynamic_axis["outputs"])
    if any(
        len(x) > 0 for x in (dynamic_axis["inputs"] + dynamic_axis["outputs"])
    ):
        return dynamic_axis
    return None


def extract_info_from_np_data(
    onnx_model: str,
    data: List[Tuple[Tuple[np.ndarray, ...], np.ndarray]],
    dynamic_axis: Dict,
    device: Device,
):
    from nebullvm.tools.utils import ifnone

    input_row = data[0][0]
    batch_size = int(input_row[0].shape[0])
    if not all([input_row[0].shape[0] == x.shape[0] for x in input_row]):
        logger.warning("Detected not consistent batch size in the inputs.")

    input_sizes = [tuple(x.shape) for x in input_row]
    input_types = [
        "int32"
        if x.dtype is np.int32
        else "int64"
        if x.dtype is np.int64
        else "float32"
        for x in input_row
    ]
    dynamic_axis = ifnone(
        dynamic_axis,
        _extract_dynamic_axis(onnx_model, data, input_sizes, device),
    )
    return batch_size, input_sizes, input_types, dynamic_axis


def get_output_sizes_onnx(
    onnx_model: str, input_tensors: List[np.ndarray], device
) -> List[Tuple[int, ...]]:
    res = run_onnx_model(onnx_model, input_tensors, device)
    sizes = [tuple(output.shape) for output in res]
    return sizes


def create_model_inputs_onnx(input_infos: List[InputInfo]) -> List[np.ndarray]:
    input_tensors = (
        np.random.randn(*input_info.size).astype(np.float32)
        if input_info.dtype is DataType.FLOAT32
        else np.random.randint(
            size=input_info.size,
            low=input_info.min_value or 0,
            high=input_info.max_value or 100,
        )
        for input_info in input_infos
    )
    return list(input_tensors)


def onnx_is_gpu_available():
    return ort.get_device() == "GPU"
