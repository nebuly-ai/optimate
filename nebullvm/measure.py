import time
from typing import Tuple, List, Union

import tensorflow as tf
import torch

from nebullvm.inference_learners.base import BaseInferenceLearner


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
    optimized_model: BaseInferenceLearner, steps: int = 100
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
    for _ in range(steps):
        starting_time = time.time()
        _ = optimized_model.predict(*model_inputs)
        latencies.append(time.time() - starting_time)
    return sum(latencies) / steps
