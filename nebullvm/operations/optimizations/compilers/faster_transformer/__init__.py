from copy import deepcopy
from typing import Union

from nebullvm.core.models import QuantizationType, DeviceType
from nebullvm.operations.optimizations.compilers.faster_transformer.bert import (  # noqa: E501
    detect_and_swap_bert_model,
)
from nebullvm.operations.optimizations.compilers.torchscript import (
    TorchScriptCompiler,
)
from nebullvm.operations.optimizations.compilers.utils import (
    get_faster_transformer_repo_path,
)
from nebullvm.optional_modules.torch import (
    GraphModule,
    Module,
    ScriptModule,
    torch,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.huggingface import PyTorchTransformerWrapper

default_lib_path = str(
    get_faster_transformer_repo_path()
    / "build"
    / "lib"
    / "libth_transformer.so"
)


def detect_and_swap_model(model, data_type="fp16", remove_padding=False):
    """currently only supports:
    - BertModel and model with BertModel as .bert attribute
    """
    model = detect_and_swap_bert_model(
        model,
        data_type=data_type,
        lib_path=default_lib_path,
        remove_padding=remove_padding,
    )
    if data_type == "fp16":
        model.half()
    elif data_type == "bf16":
        model.bfloat16()
    return model


class FasterTransformerCompiler(TorchScriptCompiler):
    supported_ops = {
        "cpu": [None, QuantizationType.STATIC, QuantizationType.DYNAMIC],
        "gpu": [
            None,
            QuantizationType.HALF,
        ],
    }

    @torch.no_grad()
    def _compile_model(
        self,
        model: Union[Module, GraphModule],
        input_data: DataManager,
        quantization_type: QuantizationType,
    ) -> ScriptModule:
        model = deepcopy(model)  # Some operations modify the model in-place
        if isinstance(model, PyTorchTransformerWrapper):
            # .core_model is a huggingface model
            data_type = (
                "fp16"
                if quantization_type is QuantizationType.HALF
                else "fp32"
            )
            model.core_model = detect_and_swap_model(
                model.core_model, data_type=data_type, remove_padding=False
            )
            if self.device.type is DeviceType.GPU:
                model.cuda()

        return super()._compile_model(model, input_data, quantization_type)
