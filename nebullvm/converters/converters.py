from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Tuple

import tensorflow as tf
from torch.nn import Module

from nebullvm.converters.tensorflow_converters import (
    convert_tf_to_onnx,
    convert_keras_to_onnx,
)
from nebullvm.converters.torch_converters import convert_torch_to_onnx


class BaseConverter(ABC):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "temp"

    @abstractmethod
    def convert(
        self, model: Any, input_size: Tuple[int, ...], save_path: Path
    ):
        raise NotImplementedError


class ONNXConverter(BaseConverter):
    ONNX_MODEL_EXTENSION = ".onnx"

    def convert(
        self, model: Any, input_size: Tuple[int, ...], save_path: Path
    ):
        onnx_name = f"{self.model_name}{self.ONNX_MODEL_EXTENSION}"
        if isinstance(model, Module):
            convert_torch_to_onnx(
                torch_model=model,
                input_size=input_size,
                output_file_path=save_path / onnx_name,
            )
            return save_path / onnx_name
        elif isinstance(model, tf.Module):
            convert_tf_to_onnx(
                model=model,
                input_size=input_size,
                output_file_path=save_path / onnx_name,
            )
        elif isinstance(model, tf.keras.Model):
            convert_keras_to_onnx(
                model=model,
                input_size=input_size,
                output_file_path=save_path / onnx_name,
            )
        else:
            raise NotImplementedError(
                f"The ONNX conversion from {type(model)} hasn't "
                f"been implemented yet!"
            )
