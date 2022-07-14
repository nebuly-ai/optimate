from typing import Any, List, Tuple, Dict

from nebullvm.measure import compute_relative_difference, compute_accuracy_drop
import warnings

QUANTIZATION_METRIC_MAP = {
    "accuracy": compute_accuracy_drop,
    "precision": compute_relative_difference,
}


def check_inputs(
    input_data: Any, batch_size: int, input_sizes: List[Tuple[int, ...]]
):
    if input_data is None and input_sizes is None:
        raise ValueError(
            "Non-admissible input detected. You need to specify either the "
            "input data or directly the input sizes."
        )
    if batch_size is not None:
        warnings.warn(
            "Batch size will be deprecated from the next release, please include "
            "it directly inside the input sizes."
        )
        input_sizes = [(batch_size,) + input_size for input_size in input_sizes]
    return input_sizes


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
