import time
from typing import Tuple, List, Union, Any

import numpy as np
from loguru import logger

from nebullvm.config import ONNX_PROVIDERS
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch, Module
from nebullvm.tools.base import Device, DeviceType
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import (
    convert_to_numpy,
    get_input_names,
    get_output_names,
)


def compute_torch_latency(
    xs: List[Tuple[torch.Tensor]],
    model: Module,
    device: Device,
    steps: int = 100,
    warmup_steps: int = 10,
) -> Tuple[float, List[float]]:
    """Compute the latency associated with the torch model.

    Args:
        xs (List[Tuple[torch.Tensor]]): List of tuples containing the
            input tensors (a single batch for the model).
        model (Module): Torch model.
        device (Device): Device where computing the latency.
        steps (int, optional): Number of input data to be used to compute the
            latency of the model. It must be a number <= len(xs). Default: 100.
        warmup_steps (int, optional): Number of input data to be used to warm
            up the model. It must be a number <= len(xs). Default: 10.

    Returns:
        Float: Average latency.
        List[Float]: List of latencies obtained.
    """
    xs = [
        tuple(t.to(device.to_torch_format()) for t in tensors)
        for tensors in xs
    ]
    model = model.to(device.to_torch_format()).eval()
    latencies = []
    with torch.no_grad():
        for i in range(warmup_steps):
            _ = model.forward(*xs[i])
        for i in range(steps):
            starting_time = time.time()
            _ = model.forward(*xs[i])
            latencies.append(time.time() - starting_time)
        latency = np.mean(latencies)
    return latency, latencies


def compute_tf_latency(
    xs: List[Tuple[tf.Tensor]],
    model: Union[tf.Module, tf.keras.Model],
    device: Device,
    steps: int = 100,
    warmup_steps: int = 10,
) -> Tuple[float, List[float]]:
    """Compute the latency associated with the tensorflow model.

    Args:
        xs (List[Tuple[tf.Tensor]]): List of tuples containing the
            input tensors (a single batch for the model).
        model (Module or keras.Model): TF model.
        device (Device): Device where computing the latency.
        steps (int, optional): Number of input data to be used to compute the
            latency of the model. It must be a number <= len(xs). Default: 100.
        warmup_steps (int, optional): Number of input data to be used to warm
            up the model. It must be a number <= len(xs). Default: 10.

    Returns:
        Float: Average latency.
        List[Float]: List of latencies obtained.
    """
    latencies = []
    with tf.device(device.to_tf_format()):
        for i in range(warmup_steps):
            _ = model(xs[i])
        for i in range(steps):
            starting_time = time.time()
            _ = model(xs[i])
            latencies.append(time.time() - starting_time)
        latency = np.mean(latencies)
        return latency, latencies


def compute_onnx_latency(
    xs: List[Tuple[np.array]],
    model: str,
    device: Device,
    steps: int = 100,
    warmup_steps: int = 10,
) -> Tuple[float, List[float]]:
    """Compute the latency associated with the ONNX model.

    Args:
        xs (List[Tuple[np.array]]): List of tuples containing the
            inputs (a single batch for the model).
        model (str): ONNX model path.
        device (Device): Device where computing the latency.
        steps (int, optional): Number of input data to be used to compute the
            latency of the model. It must be a number <= len(xs). Default: 100.
        warmup_steps (int, optional): Number of input data to be used to warm
            up the model. It must be a number <= len(xs). Default: 10.

    Returns:
        Float: Average latency.
        List[Float]: List of latencies obtained.
    """
    from nebullvm.optional_modules.onnxruntime import onnxruntime as ort

    input_names = get_input_names(model)
    output_names = get_output_names(model)

    if device.type is DeviceType.GPU:
        ONNX_PROVIDERS["cuda"][1] = (
            "CUDAExecutionProvider",
            {
                "device_id": device.idx,
            },
        )

    model = ort.InferenceSession(
        model,
        providers=ONNX_PROVIDERS["cuda"][1:]
        if device.type is DeviceType.GPU
        else ONNX_PROVIDERS["cpu"],
    )

    latencies = []
    for i in range(warmup_steps):
        inputs = {name: array for name, array in zip(input_names, xs[i])}
        _ = model.run(output_names=output_names, input_feed=inputs)
    for i in range(steps):
        inputs = {name: array for name, array in zip(input_names, xs[i])}
        starting_time = time.time()
        _ = model.run(output_names=output_names, input_feed=inputs)
        latencies.append(time.time() - starting_time)
    latency = np.mean(latencies)
    return latency, latencies


def compute_optimized_running_time(
    optimized_model: BaseInferenceLearner,
    input_data: DataManager,
    steps: int = 100,
    min_steps: int = 5,
    warmup_steps: int = 10,
) -> float:
    """Compute the running time of the optimized model.

    Args:
        optimized_model (BaseInferenceLearner): Optimized model.
        input_data: (DataManager): Dataset used to compute latency.
        steps (int, optional): Number of input data to be used to
            compute the latency of the model. Default: 100.
        min_steps (int, optional): Minimum number of iterations to
            be performed. Default: 5.
        warmup_steps (int, optional): Number of input data to be used
            to warm up the model. Default: 10.

    Returns:
        Float: Average latency.
    """

    latencies = []
    last_median = None

    # Warmup
    inputs_list = input_data.get_split("test").get_list(warmup_steps)
    for model_inputs in inputs_list:
        _ = optimized_model(*model_inputs)

    # Compute latency
    inputs_list = input_data.get_split("test").get_list(steps)
    for model_inputs in inputs_list:
        starting_time = time.time()
        _ = optimized_model(*model_inputs)
        latencies.append(time.time() - starting_time)
        if len(latencies) > min_steps:
            median = np.median(latencies)
            diff = (
                np.abs(median - last_median) / last_median
                if last_median is not None
                else 1.0
            )
            if diff < 0.05:
                return median
            last_median = median
    return np.median(latencies)


def compute_relative_difference(
    tensor_1: Any,
    tensor_2: Any,
    y: Any = None,
    eps: float = 1e-5,
) -> float:
    if y is not None:
        logger.debug(
            "Received a label for the precision computation. "
            "It will be ignored."
        )

    tensor_1, tensor_2 = map(convert_to_numpy, (tensor_1, tensor_2))

    assert tensor_1.shape == tensor_2.shape, (
        "The outputs of the original and optimized models have "
        "different shapes"
    )

    diff = np.abs(tensor_1 - tensor_2) / (
        np.maximum(np.abs(tensor_1), np.abs(tensor_2)) + eps
    )
    return float(np.mean(diff))


def compute_accuracy_drop(tensor_1: Any, tensor_2: Any, y: Any) -> float:
    assert y is not None, (
        "No label found in the dataloader provided. "
        "To use accuracy metric, you must set also the labels"
    )
    tensor_1, tensor_2, y = map(convert_to_numpy, (tensor_1, tensor_2, y))
    accuracy_1 = np.mean(tensor_1.argmax(axis=-1) == y)
    accuracy_2 = np.mean(tensor_2.argmax(axis=-1) == y)
    return accuracy_1 - accuracy_2


QUANTIZATION_METRIC_MAP = {
    "accuracy": compute_accuracy_drop,
    "numeric_precision": compute_relative_difference,
}
