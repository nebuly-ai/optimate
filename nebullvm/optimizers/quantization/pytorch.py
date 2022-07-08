import copy
from typing import List, Tuple

import torch
from torch.ao.quantization.stubs import QuantStub, DeQuantStub

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


def _quantize_dynamic(model: torch.nn.Module):
    layer_types = {
        type(layer)
        for layer in model.children()
        if len(list(layer.parameters())) > 0
    }
    quantized_model = torch.quantization.quantize_dynamic(
        model=model, qconfig_spec=layer_types, dtype=torch.qint8
    )
    return quantized_model


def _quantize_static(
    model: torch.nn.Module, input_data: List[Tuple[torch.Tensor, ...]]
):
    model = _QuantWrapper(model)
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model = torch.quantization.fuse_modules(model, [["conv", "relu"]])
    model = torch.quantization.prepare(model)
    for tensors in input_data:
        _ = model(*tensors)
    return torch.quantization.convert(model)


def _half_precision(model: torch.nn.Module):
    return model.half()


def quantize_torch(
    model: torch.nn.Module,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    input_data_torch: List[Tuple[torch.Tensor, ...]],
):
    model = copy.deepcopy(model)
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
