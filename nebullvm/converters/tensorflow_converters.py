from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Union, Tuple, List

import tensorflow as tf
import tf2onnx


def get_outputs_sizes_tf(
    tf_model: Union[tf.Module, tf.keras.Model], input_tensors: List[tf.Tensor]
) -> List[Tuple[int, ...]]:
    outputs = tf_model(*input_tensors)
    if isinstance(outputs, tf.Tensor):
        return [tuple(tf.shape(outputs))]
    return [tuple(x.size()) for x in outputs]


def convert_tf_to_onnx(
    model: tf.Module,
    output_file_path: Union[str, Path],
):
    """Convert TF models into ONNX.

    Args:
        model (tf.Module): TF model.
        input_sizes (List[tuple]): Sizes of the model's input tensors.
        output_file_path (Path): Path where storing the output file.
    """
    with TemporaryDirectory() as temp_dir:
        tf.saved_model.save(model, export_dir=temp_dir)
        onnx_cmd = [
            "python3",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            f"{temp_dir}",
            "--output",
            f"{output_file_path}",
            "--opset",
            "11",
        ]
        subprocess.run(onnx_cmd)


def convert_keras_to_onnx(
    model: tf.keras.Model,
    input_sizes: List[Tuple[int, ...]],
    output_file_path: Union[str, Path],
):
    """Convert keras models into ONNX.

    Args:
        model (tf.Module): keras model.
        input_sizes (List[tuple]): Sizes of the model's input tensors.
        output_file_path (Path): Path where storing the output file.
    """
    spec = (
        tf.TensorSpec(input_size, tf.float32, name=f"input_{i}")
        for i, input_size in enumerate(input_sizes)
    )
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=11, output_path=output_file_path
    )
