from typing import Union, List, Tuple, Any, Optional, Dict

import numpy as np
from loguru import logger

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.tools.base import InputInfo, DataType, Device


def get_outputs_sizes_tf(
    tf_model: Union[tf.Module, tf.keras.Model],
    input_tensors: List[tf.Tensor],
    device: Device,
) -> List[Tuple[int, ...]]:
    with tf.device(device.to_tf_format()):
        outputs = tf_model(input_tensors)
    if isinstance(outputs, tf.Tensor) and outputs is not None:
        return [tuple(outputs.shape)]
    return [tuple(x.shape) for x in outputs]


def create_model_inputs_tf(input_infos: List[InputInfo]) -> List[tf.Tensor]:
    return [
        tf.random_normal_initializer()(
            shape=(
                input_info.size[0],
                *input_info.size[2:],
                input_info.size[1],
            )
        )
        if input_info.dtype is DataType.FLOAT32
        else tf.random.uniform(
            shape=(
                input_info.size[0],
                *input_info.size[2:],
                input_info.size[1],
            ),
            minval=input_info.min_value or 0,
            maxval=input_info.max_value or 100,
            dtype=tf.int32,
        )
        for input_info in input_infos
    ]


def run_tf_model(
    model: tf.Module,
    input_tensors: Tuple[tf.Tensor],
    device: Device,
) -> Tuple[tf.Tensor]:
    with tf.device(device.to_tf_format()):
        pred = model(input_tensors)
    if isinstance(pred, tf.Tensor):
        pred = (pred,)
    return pred


def _extract_dynamic_axis(
    tf_model: tf.Module,
    dataset: List[Tuple[Tuple[tf.Tensor, ...], Any]],
    input_sizes: List[Tuple[int, ...]],
    device: Device,
    max_data: int = 100,
) -> Optional[Dict]:
    from nebullvm.tools.utils import inspect_dynamic_size

    dynamic_axis = {"inputs": [{}] * len(input_sizes), "outputs": []}
    output_sizes = []
    for i, input_data in enumerate(dataset):
        input_tensors = input_data[0]
        if i >= max_data:
            break
        inspect_dynamic_size(
            input_tensors, input_sizes, dynamic_axis["inputs"]
        )
        outputs = tuple(run_tf_model(tf_model, input_tensors, device))
        if i == 0:
            dynamic_axis["outputs"] = [{}] * len(outputs)
            output_sizes = [tuple(output.shape[1:]) for output in outputs]
        inspect_dynamic_size(outputs, output_sizes, dynamic_axis["outputs"])
    if any(
        len(x) > 0 for x in (dynamic_axis["inputs"] + dynamic_axis["outputs"])
    ):
        return dynamic_axis
    return None


def extract_info_from_tf_data(
    tf_model: tf.Module,
    dataset: List[Tuple[Tuple[tf.Tensor, ...], Any]],
    dynamic_axis: Dict,
    device: Device,
):
    from nebullvm.tools.utils import ifnone

    input_row = dataset[0][0]
    batch_size = int(input_row[0].shape[0])
    if not all([input_row[0].shape[0] == x.shape[0] for x in input_row]):
        logger.warning("Detected not consistent batch size in the inputs.")

    input_sizes = [tuple(x.shape) for x in input_row]
    input_types = [
        "int32"
        if x.dtype in [tf.int32, np.int32]
        else "int64"
        if x.dtype in [tf.int64, np.int64]
        else "float32"
        for x in input_row
    ]

    dynamic_axis = ifnone(
        dynamic_axis,
        _extract_dynamic_axis(tf_model, dataset, input_sizes, device),
    )
    return batch_size, input_sizes, input_types, dynamic_axis


def tensorflow_is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0


def tensorflow_get_gpu_name():
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        details.get("device_name", "Unknown GPU")
        return details["device_name"]
    else:
        return "Unknown GPU"
