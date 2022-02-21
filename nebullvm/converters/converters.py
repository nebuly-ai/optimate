from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Tuple, List

import tensorflow as tf
from torch.nn import Module

from nebullvm.converters.tensorflow_converters import (
    convert_tf_to_onnx,
    convert_keras_to_onnx,
)
from nebullvm.converters.torch_converters import convert_torch_to_onnx


class BaseConverter(ABC):
    """Base class for converters.

    Attributes:
        model_name (str, optional): name of the model. If not given 'temp' will
            be used as model name.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "temp"

    @abstractmethod
    def convert(
        self, model: Any, input_size: Tuple[int, ...], save_path: Path
    ):
        raise NotImplementedError


class ONNXConverter(BaseConverter):
    """Class for converting models from a supported framework to ONNX.

    Attributes:
        model_name (str, optional): name of the model. If not given 'temp' will
            be used as model name.
    """

    ONNX_MODEL_EXTENSION = ".onnx"

    def convert(
        self, model: Any, input_sizes: List[Tuple[int, ...]], save_path: Path
    ):
        """Convert the input model in ONNX.

        Args:
            model (any, optional): Model to be converted. The model can be in
                either the tensorflow or pytorch framework.
            input_sizes (List[Tuple[int, ...]]): Size of the input data.
            save_path (Path): Path to the directory where saving the onnx
                model.

        Returns:
            Path: Path to the onnx file.
        """
        onnx_name = f"{self.model_name}{self.ONNX_MODEL_EXTENSION}"
        if isinstance(model, Module):
            convert_torch_to_onnx(
                torch_model=model,
                input_sizes=input_sizes,
                output_file_path=save_path / onnx_name,
            )
            return save_path / onnx_name
        elif isinstance(model, tf.Module):
            convert_tf_to_onnx(
                model=model,
                output_file_path=save_path / onnx_name,
            )
            return save_path / onnx_name
        elif isinstance(model, tf.keras.Model):
            convert_keras_to_onnx(
                model=model,
                input_sizes=input_sizes,
                output_file_path=save_path / onnx_name,
            )
            return save_path / onnx_name
        else:
            raise NotImplementedError(
                f"The ONNX conversion from {type(model)} hasn't "
                f"been implemented yet!"
            )
