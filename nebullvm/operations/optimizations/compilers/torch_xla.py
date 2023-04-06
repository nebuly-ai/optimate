from nebullvm.core.models import QuantizationType
from nebullvm.operations.optimizations.compilers.torchscript import (
    TorchScriptCompiler,
)
from nebullvm.optional_modules.torch import (
    torch,
)
from nebullvm.tools.data import DataManager


class TorchXLACompiler(TorchScriptCompiler):
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
