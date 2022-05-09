import warnings
from typing import Tuple, Callable, Any, List

import numpy as np

from nebullvm.base import QuantizationType
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.measure import compute_relative_difference


def check_precision(
    optimized_learner: BaseInferenceLearner,
    input_data: List[Tuple[Any, ...]],
    base_outputs_list: List[Tuple[Any, ...]],
    perf_loss_ths: float,
    metric_func: Callable = None,
    ys: List = None,
    aggregation_func: Callable = np.mean,
) -> bool:
    metric_func = metric_func or compute_relative_difference
    relative_differences = []
    if ys is None:
        ys = [None] * len(input_data)
    for inputs, base_outputs, y in zip(input_data, base_outputs_list, ys):
        opt_outputs = optimized_learner(*inputs)
        relative_difference = max(
            metric_func(base_output, opt_output, y)
            for base_output, opt_output in zip(base_outputs, opt_outputs)
        )
        relative_differences.append(relative_difference)
    relative_difference = aggregation_func(relative_differences)
    return relative_difference <= perf_loss_ths


def check_quantization(
    quantization_type: QuantizationType, perf_loss_ths: float
):
    if quantization_type is None and perf_loss_ths is not None:
        raise ValueError(
            "When a quantization threshold is given it is necessary to "
            "specify the quantization algorithm too."
        )
    if quantization_type is not None and perf_loss_ths is None:
        warnings.warn(
            "Got a valid quantization type without any given quantization "
            "threshold. The quantization step will be ignored."
        )
