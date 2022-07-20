import logging
import time
from typing import Tuple, List, Union, Any

import numpy as np
import tensorflow as tf
import torch

from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.utils.onnx import convert_to_numpy


def compute_torch_latency(
    xs: List[torch.Tensor],
    model: torch.nn.Module,
    device: str,
    steps: int,
) -> Tuple[float, List[float]]:
    """Compute the latency associated with the torch model.

    Args:
        xs (List[Tensor]): List of input tensors (a single batch for the model)
        model (Module): Torch model.
        device (str): Device where computing the latency.
        steps (int): Number of times the experiment needs to be performed for
            computing the statistics.

    Returns:
        Float: Average latency.
        List[Float]: List of latencies obtained.
    """
    xs = [x.to(device) for x in xs]
    model = model.to(device)
    latencies = []
    for _ in range(steps):
        starting_time = time.time()
        _ = model.forward(*xs)
        latencies.append(time.time() - starting_time)
    latency = sum(latencies) / steps
    return latency, latencies


def compute_tf_latency(
    xs: List[tf.Tensor],
    model: Union[tf.Module, tf.keras.Model],
    device: str,
    steps: int,
) -> Tuple[float, List[float]]:
    """Compute the latency associated with the tensorflow model.

    Args:
        xs (List[Tensor]): List of input tensors (a single batch for the model)
        model (Module or keras.Model): TF model.
        device (str): Device where computing the latency.
        steps (int): Number of times the experiment needs to be performed for
            computing the statistics.

    Returns:
        Float: Average latency.
        List[Float]: List of latencies obtained.
    """
    latencies = []
    with tf.device(device):
        for _ in range(steps):
            starting_time = time.time()
            _ = model(*xs)
            latencies.append(time.time() - starting_time)
        latency = sum(latencies) / steps
        return latency, latencies


def compute_optimized_running_time(
    optimized_model: BaseInferenceLearner, steps: int = 100, min_steps=5
) -> float:
    """Compute the running time of the optimized model.

    Args:
        optimized_model (BaseInferenceLearner): Optimized model.
        steps (int): Number of times the experiment needs to be performed for
            computing the statistics.

    Returns:
        Float: Average latency.
    """

    model_inputs = optimized_model.get_inputs_example()

    latencies = []
    last_median = None
    for _ in range(steps):
        starting_time = time.time()
        _ = optimized_model.predict(*model_inputs)
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
        logging.debug(
            "Received a label for the precision computation. "
            "It will be ignored."
        )
    tensor_1, tensor_2 = map(convert_to_numpy, (tensor_1, tensor_2))
    diff = np.abs(tensor_1 - tensor_2) / (
        np.maximum(np.abs(tensor_1), np.abs(tensor_2)) + eps
    )
    return np.max(diff)


def compute_accuracy_drop(tensor_1: Any, tensor_2: Any, y: Any) -> float:
    tensor_1, tensor_2, y = map(convert_to_numpy, (tensor_1, tensor_2, y))
    accuracy_1 = np.mean(tensor_1.argmax(axis=-1) == y)
    accuracy_2 = np.mean(tensor_2.argmax(axis=-1) == y)
    return accuracy_1 - accuracy_2
