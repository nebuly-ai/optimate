from typing import Union, List, Tuple

from nebullvm.base import InputInfo, DataType
from nebullvm.optional_modules.tensorflow import tensorflow as tf


def get_outputs_sizes_tf(
    tf_model: Union[tf.Module, tf.keras.Model],
    input_tensors: List[tf.Tensor],
    device: str,
) -> List[Tuple[int, ...]]:
    with tf.device(device):
        outputs = tf_model(*input_tensors)
    if isinstance(outputs, tf.Tensor) and outputs is not None:
        return [tuple(outputs.shape)]
    return [tuple(x.shape) for x in outputs]


def create_model_inputs_tf(
    batch_size: int, input_infos: List[InputInfo]
) -> List[tf.Tensor]:
    return [
        tf.random_normal_initializer()(
            shape=(batch_size, *input_info.size[1:], input_info.size[0])
        )
        if input_info.dtype is DataType.FLOAT
        else tf.random.uniform(
            shape=(batch_size, *input_info.size[1:], input_info.size[0]),
            minval=input_info.min_value or 0,
            maxval=input_info.max_value or 100,
            dtype=tf.int32,
        )
        for input_info in input_infos
    ]


def run_tf_model(
    model: tf.Module,
    input_tensors: Tuple[tf.Tensor],
    device: str,
) -> Tuple[tf.Tensor]:
    with tf.device(device):
        pred = model.predict(input_tensors)
    if isinstance(pred, tf.Module) and pred is not None:
        pred = (pred,)
    return pred


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
