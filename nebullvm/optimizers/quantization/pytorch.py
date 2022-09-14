import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch.fx import symbolic_trace
from torch.quantization import default_dynamic_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from nebullvm.base import QuantizationType
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.transformations.precision_tfms import HalfPrecisionTransformation


class _QuantWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(_QuantWrapper, self).__init__()
        qconfig = model.qconfig if hasattr(model, "qconfig") else None
        self.quant = QuantStub(qconfig)
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, *inputs: torch.Tensor):
        inputs = (self.quant(x) for x in inputs)
        outputs = self.model(*inputs)
        return tuple(self.dequant(x) for x in outputs)


def _quantize_dynamic_torch(model: torch.nn.Module):
    layer_types = {
        type(layer)
        for layer in model.children()
        if len(list(layer.parameters())) > 0
    }
    return torch.quantization.quantize_dynamic(
        model=model, qconfig_spec=layer_types, dtype=torch.qint8
    )


def _quantize_dynamic_torch_fx(model: torch.fx.GraphModule):
    qconfig_dict = {"": default_dynamic_qconfig}
    model_prepared = prepare_fx(model, qconfig_dict)
    return convert_fx(model_prepared)


def _quantize_static_torch(
    model: torch.nn.Module,
    input_data: List[Tuple[torch.Tensor, ...]],
    backend: str,
):
    model = _QuantWrapper(model)
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    # TODO: change line below, it's wrong
    # model = torch.quantization.fuse_modules(model, [["conv", "relu"]])
    model = torch.quantization.prepare(model)
    with torch.no_grad():
        for tensors in input_data:
            _ = model(*tensors)
    return torch.quantization.convert(model)


def _quantize_static_torch_fx(
    model: torch.fx.GraphModule,
    input_data: List[Tuple[torch.Tensor, ...]],
    backend: str,
):
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    model = prepare_fx(model, qconfig_dict)
    with torch.no_grad():
        for tensors in input_data:
            _ = model(*tensors)
    return convert_fx(model)


def _quantize_static(
    model: Union[torch.nn.Module, torch.fx.GraphModule],
    input_data: List[Tuple[torch.Tensor, ...]],
):
    if torch.cuda.is_available():
        raise AssertionError("Quantization for torch is only available on CPU")

    backend = (
        "fbgemm"
        if "fbgemm" in torch.backends.quantized.supported_engines
        else "qnnpack"
    )

    if isinstance(model, torch.fx.GraphModule):
        return _quantize_static_torch_fx(model, input_data, backend)
    else:
        return _quantize_static_torch(model, input_data, backend)


def _quantize_dynamic(model: Union[torch.nn.Module, torch.fx.GraphModule]):
    if torch.cuda.is_available():
        raise AssertionError("Quantization for torch is only available on CPU")

    if isinstance(model, torch.fx.GraphModule):
        return _quantize_dynamic_torch(model)
    else:
        return _quantize_dynamic_torch(model)


def _half_precision(model: torch.nn.Module):
    return model.half()


def quantize_torch(
    model: torch.nn.Module,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    input_data_torch: List[Tuple[torch.Tensor, ...]],
):
    model = copy.deepcopy(model).eval()

    try:
        model = symbolic_trace(model)
    except Exception:
        warnings.warn("Unable to trace model with torch.fx")

    if quantization_type is QuantizationType.HALF:
        input_tfms.append(HalfPrecisionTransformation())
        return _half_precision(model), input_tfms
    elif quantization_type is QuantizationType.STATIC:
        return _quantize_static(model, input_data_torch), input_tfms
    elif quantization_type is QuantizationType.DYNAMIC:
        return _quantize_dynamic(model), input_tfms
    else:
        raise NotImplementedError(
            f"No quantization implemented for quantization "
            f"type {quantization_type}"
        )
