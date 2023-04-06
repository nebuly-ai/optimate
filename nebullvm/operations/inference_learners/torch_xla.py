import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Union

from nebullvm.core.models import Device, DeviceType, ModelParams
from nebullvm.operations.inference_learners.base import (
    PytorchBaseInferenceLearner,
    LearnerMetadata,
)
from nebullvm.optional_modules.torch import (
    torch,
)
from nebullvm.tools.transformations import MultiStageTransformation


class TorchXLAInferenceLearner(PytorchBaseInferenceLearner):
    MODEL_NAME = "model_scripted.pt"
    name = "TorchXLA"

    def __init__(self, torch_model: torch.nn.Module, device: Device, **kwargs):
        super().__init__(**kwargs)
        self.model = torch_model.eval()
        if device.type is DeviceType.TPU:
            self.model.to(device.to_torch_format())
        self.device = device
        self._is_gpu_ready = self.device.type is DeviceType.TPU

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.device.type is DeviceType.TPU and not self._is_gpu_ready:
            self.set_model_on_gpu()
        if self.device.type is DeviceType.TPU:
            input_tensors = (
                t.to(self.device.to_torch_format()) for t in input_tensors
            )
        with torch.no_grad():
            res = self.model(*input_tensors)
            if not isinstance(res, tuple):
                return (res,)
            return tuple(out for out in res)

    def get_size(self):
        try:
            if hasattr(self.model, "core_model"):
                return len(pickle.dumps(self.model.core_model, -1))
            else:
                # Normal torch model
                return len(pickle.dumps(self.model, -1))
        except RuntimeError:
            with TemporaryDirectory() as tmp_dir:
                self.save(tmp_dir)
                return sum(
                    os.path.getsize(Path(tmp_dir) / f)
                    for f in os.listdir(Path(tmp_dir))
                    if os.path.isfile(Path(tmp_dir) / f)
                )

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        self.model.cpu()
        torch.save(self.model, path / self.MODEL_NAME)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        model = torch.load(path / cls.MODEL_NAME)
        metadata = LearnerMetadata.read(path)
        device = Device.from_str(metadata.device)
        model.to(device.to_torch_format())
        return cls(
            torch_model=model,
            network_parameters=ModelParams(**metadata.network_parameters),
            input_tfms=MultiStageTransformation.from_dict(metadata.input_tfms)
            if metadata.input_tfms is not None
            else None,
            device=device,
        )
