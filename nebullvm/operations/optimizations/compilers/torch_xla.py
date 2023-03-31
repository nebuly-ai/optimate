from nebullvm.operations.optimizations.compilers.pytorch import (
    PytorchBackendCompiler,
)
from nebullvm.optional_modules.torch import (
    torch,
)
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager


class TorchXLACompiler(PytorchBackendCompiler):
    supported_ops = {
        "cpu": [],
        "gpu": [],
        "tpu": [None, QuantizationType.HALF],
    }

    @torch.no_grad()
    def _compile_model(
        self,
        model: torch.nn.Module,
        input_data: DataManager,
        quantization_type: QuantizationType,
    ) -> torch.nn.Module:
        compiled_model = model.to(self.device.to_torch_format())
        return compiled_model
