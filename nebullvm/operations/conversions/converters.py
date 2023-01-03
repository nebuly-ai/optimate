import abc
from pathlib import Path
from typing import Optional, List, Union

from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.pytorch import convert_torch_to_onnx
from nebullvm.operations.conversions.tensorflow import convert_tf_to_onnx
from nebullvm.optional_modules.onnx import onnx
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import DeepLearningFramework, ModelParams
from nebullvm.tools.data import DataManager


class Converter(Operation, abc.ABC):
    ONNX_EXTENSION = ".onnx"
    TORCH_EXTENSION = ".pt"
    TF_EXTENSION = ".pb"

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.model = None
        self.data = None
        self.converted_models = None
        self.model_params = None
        self.device = None
        self.model_name = model_name or "temp"

    def set_state(
        self, model: Union[torch.nn.Module, tf.Module, str], data: DataManager
    ):
        self.model = model
        self.data = data
        return self

    def get_result(self) -> List:
        return self.converted_models


class PytorchConverter(Converter):
    DEST_FRAMEWORKS = [DeepLearningFramework.NUMPY]

    def execute(
        self,
        save_path: Path,
        model_params: ModelParams,
    ):
        self.converted_models = [self.model]
        for framework in self.DEST_FRAMEWORKS:
            if framework is DeepLearningFramework.NUMPY:
                self.onnx_conversion(save_path, model_params)
            else:
                raise NotImplementedError()

    def onnx_conversion(self, save_path, model_params):
        onnx_path = save_path / f"{self.model_name}{self.ONNX_EXTENSION}"
        onnx_model_path = convert_torch_to_onnx(
            torch_model=self.model,
            input_data=self.data,
            model_params=model_params,
            output_file_path=onnx_path,
            device=self.device,
        )
        if self.converted_models is None:
            self.converted_models = [onnx_model_path]
        else:
            self.converted_models.append(onnx_model_path)

    def tensorflow_conversion(self):
        # TODO: Implement conversion from Pytorch to Tensorflow
        raise NotImplementedError()


class TensorflowConverter(Converter):
    DEST_FRAMEWORKS = [DeepLearningFramework.NUMPY]

    def execute(
        self,
        save_path: Path,
        model_params: ModelParams,
    ):
        self.converted_models = [self.model]
        for framework in self.DEST_FRAMEWORKS:
            if framework is DeepLearningFramework.NUMPY:
                self.onnx_conversion(save_path, model_params)
            else:
                raise NotImplementedError()

    def onnx_conversion(self, save_path, model_params):
        onnx_path = save_path / f"{self.model_name}{self.ONNX_EXTENSION}"
        onnx_model_path = convert_tf_to_onnx(
            model=self.model,
            model_params=model_params,
            output_file_path=onnx_path,
        )
        if self.converted_models is None:
            self.converted_models = [onnx_model_path]
        else:
            self.converted_models.append(onnx_model_path)

    def pytorch_conversion(self):
        # TODO: Implement conversion from Tensorflow to Pytorch
        raise NotImplementedError()


class ONNXConverter(Converter):
    DEST_FRAMEWORKS = []

    def execute(self, save_path, model_params):
        onnx_path = save_path / f"{self.model_name}{self.ONNX_EXTENSION}"
        try:
            model_onnx = onnx.load(str(self.model))
            onnx.save(model_onnx, str(onnx_path))
        except Exception:
            self.logger.error(
                "The provided onnx model path is invalid. Please provide"
                " a valid path to a model in order to use Nebullvm."
            )
            self.converted_models = []

        self.converted_models = [str(onnx_path)]

    def tensorflow_conversion(self):
        # TODO: Implement conversion from ONNX to Tensorflow
        raise NotImplementedError()

    def pytorch_conversion(self):
        # TODO: Implement conversion from ONNX to Pytorch
        raise NotImplementedError()
