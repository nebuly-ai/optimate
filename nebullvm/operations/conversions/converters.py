import abc
from pathlib import Path
from typing import Optional

from nebullvm.base import ModelParams, Device
from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.torch import convert_torch_to_onnx
from nebullvm.tools.base import DeepLearningFramework


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

    def set_state(self, model, data):
        self.model = model
        self.data = data
        return self

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    def is_result_available(self) -> bool:
        return self.converted_models is not None


class PytorchConverter(Converter):
    DEST_FRAMEWORKS = [DeepLearningFramework.NUMPY]

    def execute(
        self,
        save_path: Path,
        model_params: ModelParams,
        device: Device,
    ):
        for framework in self.DEST_FRAMEWORKS:
            if framework is DeepLearningFramework.NUMPY:
                self.onnx_conversion(save_path, model_params, device)
            else:
                raise NotImplementedError()

    def onnx_conversion(self, save_path, model_params, device):
        onnx_path = save_path / f"{self.model_name}{self.ONNX_EXTENSION}"
        onnx_model = convert_torch_to_onnx(
            torch_model=self.model,
            input_data=self.data,
            model_params=model_params,
            output_file_path=onnx_path,
            device=device,
        )
        if self.converted_models is None:
            self.converted_models = [onnx_model]
        else:
            self.converted_models.append(onnx_model)

    def tensorflow_conversion(self):
        raise NotImplementedError()


class TensorflowConverter(Converter):
    DEST_FRAMEWORKS = [DeepLearningFramework.NUMPY]

    def execute(self, **kwargs):
        pass

    def onnx_conversion(self):
        pass

    def pytorch_conversion(self):
        raise NotImplementedError()


class OnnxConverter(Converter):
    DEST_FRAMEWORKS = []

    def execute(self, **kwargs):
        pass

    def tensorflow_conversion(self):
        raise NotImplementedError()

    def pytorch_conversion(self):
        raise NotImplementedError()
