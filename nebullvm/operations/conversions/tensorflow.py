import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from loguru import logger

from nebullvm.config import ONNX_OPSET_VERSION
from nebullvm.optional_modules.tensorflow import tensorflow as tf, tf2onnx
from nebullvm.optional_modules.onnx import onnx
from nebullvm.tools.base import ModelParams
from nebullvm.tools.huggingface import TensorFlowTransformerWrapper


def convert_tf_to_onnx(
    model: Union[tf.Module, tf.keras.Model],
    model_params: ModelParams,
    output_file_path: Union[str, Path],
):
    """Convert TF models into ONNX.

    Args:
        model (Union[tf.Module, tf.keras.Model]): TF model.
        model_params (ModelParams): Info about model parameters.
        output_file_path (Path): Path where storing the output file.
    """

    try:
        if isinstance(model, tf.keras.Model) or (
            isinstance(model, TensorFlowTransformerWrapper)
            and isinstance(model.core_model, tf.keras.Model)
        ):
            return convert_keras_to_onnx(model, model_params, output_file_path)
        else:
            return convert_tf_saved_model_to_onnx(model, output_file_path)
    except Exception:
        logger.warning(
            "Something went wrong during conversion from tensorflow"
            " to onnx model. ONNX pipeline will be unavailable."
        )
        return None


def convert_tf_saved_model_to_onnx(
    model: tf.Module, output_file_path: Union[str, Path]
):
    """Convert TF models into ONNX.
    Args:
        model (tf.Module): TF model.
        output_file_path (Path): Path where storing the output file.
    """
    with TemporaryDirectory() as temp_dir:
        tf.saved_model.save(model, export_dir=temp_dir)

        try:
            subprocess.check_output(["python3", "--version"])
            python_cmd = "python3"
        except subprocess.CalledProcessError:
            python_cmd = "python"

        onnx_cmd = [
            python_cmd,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            f"{temp_dir}",
            "--output",
            f"{output_file_path}",
            "--opset",
            f"{ONNX_OPSET_VERSION}",
        ]
        subprocess.run(onnx_cmd)
        onnx.load(output_file_path)

    return output_file_path


def convert_keras_to_onnx(
    model: tf.keras.Model,
    model_params: ModelParams,
    output_file_path: Union[str, Path],
):
    """Convert keras models into ONNX.

    Args:
        model (tf.keras.Model): keras model.
        model_params (ModelParams): Model Parameters as input sizes and
            dynamic axis information.
        output_file_path (Path): Path where storing the output file.
    """
    # get data types for each input
    dtypes = [
        model_params.input_infos[i].dtype.value
        for i in range(len(model_params.input_infos))
    ]
    # get input shapes for each input
    shapes = [
        [int(x) for x in model_params.input_infos[i].size]
        for i in range(len(model_params.input_infos))
    ]
    # set the dynamic axes for each input
    if isinstance(model, TensorFlowTransformerWrapper):
        names = list(model.inputs_types.keys())
    else:
        names = [f"input_{i}" for i in range(len(model_params.input_infos))]

    input_signature = tuple(
        tf.TensorSpec(
            (
                None
                if model_params.dynamic_info is not None
                and dim in model_params.dynamic_info.inputs[i]
                else shape[dim]
                for dim in range(len(shape))
            ),
            dtype,
            name=name,
        )
        for i, (shape, dtype, name) in enumerate(zip(shapes, dtypes, names))
    )

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=ONNX_OPSET_VERSION,
        output_path=output_file_path,
    )

    return output_file_path
