import time
from typing import Tuple, List, Union

import tensorflow as tf
import torch

from nebullvm.inference_learners.base import BaseInferenceLearner


def compute_torch_latency(
    x: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    steps: int,
) -> Tuple[float, List[float]]:
    x = x.to(device)
    model = model.to(device)
    latencies = []
    for _ in range(steps):
        starting_time = time.time()
        _ = model.forward(x)
        latencies.append(time.time() - starting_time)
    latency = sum(latencies) / steps
    return latency, latencies


def compute_tf_latency(
    x: tf.Tensor,
    model: Union[tf.Module, tf.keras.Model],
    device: str,
    steps: int,
) -> Tuple[float, List[float]]:
    latencies = []
    with tf.device(device):
        for _ in range(steps):
            starting_time = time.time()
            _ = model(x)
            latencies.append(time.time() - starting_time)
        latency = sum(latencies) / steps
        return latency, latencies


def compute_optimized_running_time(
    optimized_model: BaseInferenceLearner, steps: int = 100
) -> float:
    model_inputs = optimized_model.get_inputs_example()
    latencies = []
    for _ in range(steps):
        starting_time = time.time()
        _ = optimized_model.predict(**model_inputs)
        latencies.append(time.time() - starting_time)
    return sum(latencies) / steps
