from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Union, Tuple

import tensorflow as tf
import tf2onnx.convert


def convert_tf_to_onnx(
    model: tf.Module,
    input_size: Tuple[int, ...],
    output_file_path: Union[str, Path],
):
    """Convert TF models into ONNX.

    Args:
        model (tf.Module): TF model.
        input_size (tuple): Size of the model's input tensors.
        output_file_path (Path): Path where storing the output file.
    """
    with TemporaryDirectory() as temp_dir:
        tf.saved_model.save(model, export_dir=temp_dir)
        onnx_cmd = (
            f"python -m tf2onnx.convert --saved-model {temp_dir/'model.tf'} "
            f"--inputs input:0{list(input_size)} "
            f"--output {output_file_path} --opset 11"
        )
        subprocess.run(onnx_cmd)


def convert_keras_to_onnx(
    model: tf.keras.Model,
    input_size: Tuple[int, ...],
    output_file_path: Union[str, Path],
):
    """Convert keras models into ONNX.

    Args:
        model (tf.Module): keras model.
        input_size (tuple): Size of the model's input tensors.
        output_file_path (Path): Path where storing the output file.
    """
    spec = (tf.TensorSpec(input_size, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=11, output_path=output_file_path
    )
