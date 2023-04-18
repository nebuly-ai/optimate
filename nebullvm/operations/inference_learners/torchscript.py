from pathlib import Path
from typing import Tuple, Union, Optional, List

from nebullvm.core.models import Device, DeviceType, ModelParams
from nebullvm.operations.inference_learners.base import (
    PytorchBaseInferenceLearner,
    LearnerMetadata,
)
from nebullvm.optional_modules.torch import (
    torch,
    symbolic_trace,
    Module,
    ScriptModule,
    GraphModule,
)
from nebullvm.tools.transformations import MultiStageTransformation


class TorchScriptInferenceLearner(PytorchBaseInferenceLearner):
    MODEL_NAME = "model_scripted.pt"
    name = "TorchScript"

    def __init__(self, torch_model: ScriptModule, device: Device, **kwargs):
        super().__init__(**kwargs)
        self.model = torch_model.eval()
        if device.type is DeviceType.GPU:
            self.model.to(device.to_torch_format())
        self.device = device
        self._is_gpu_ready = self.device.type is DeviceType.GPU

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.device.type is DeviceType.GPU and not self._is_gpu_ready:
            self.set_model_on_gpu()
        if self.device.type is DeviceType.GPU:
            input_tensors = (
                t.to(self.device.to_torch_format()) for t in input_tensors
            )
        with torch.no_grad():
            res = self.model(*input_tensors)
            if not isinstance(res, tuple):
                res = res.to(self.device.to_torch_format())
                return (res,)
            return tuple(out.to(self.device.to_torch_format()) for out in res)

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)

        torch.jit.save(self.model, path / self.MODEL_NAME)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        model = torch.jit.load(path / cls.MODEL_NAME)
        metadata = LearnerMetadata.read(path)
        device = Device.from_str(metadata.device)
        return cls(
            torch_model=model,
            network_parameters=ModelParams(**metadata.network_parameters),
            input_tfms=MultiStageTransformation.from_dict(metadata.input_tfms)
            if metadata.input_tfms is not None
            else None,
            device=device,
        )

    @classmethod
    def from_torch_model(
        cls,
        model: Union[Module, GraphModule],
        network_parameters: ModelParams,
        device: Device,
        input_tfms: Optional[MultiStageTransformation] = None,
        input_data: List[torch.Tensor] = None,
    ):
        if device.type is DeviceType.GPU:
            input_data = [t.to(device.to_torch_format()) for t in input_data]

        if not isinstance(model, torch.fx.GraphModule):
            model.eval()
            try:
                model_scripted = symbolic_trace(model)
                model_scripted = torch.jit.script(model_scripted)
            except Exception:
                try:
                    model_scripted = torch.jit.script(model)
                except Exception:
                    model_scripted = torch.jit.trace(model, tuple(input_data))
        else:
            model_scripted = torch.jit.script(model)

        return cls(
            torch_model=model_scripted,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
            input_data=input_data,
            device=device,
        )
