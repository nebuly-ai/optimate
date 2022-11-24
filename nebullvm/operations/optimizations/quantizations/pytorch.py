import copy
from typing import List, Tuple, Union

from nebullvm.operations.optimizations.quantizations.base import Quantizer
from nebullvm.optional_modules.torch import (
    torch,
    Module,
    symbolic_trace,
    QuantStub,
    DeQuantStub,
    GraphModule,
    default_dynamic_qconfig,
    prepare_fx,
    convert_fx,
)
from nebullvm.tools.base import Device, QuantizationType
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    HalfPrecisionTransformation,
)
from nebullvm.tools.utils import check_module_version


class _QuantWrapper(Module):
    def __init__(self, model: Module):
        super(_QuantWrapper, self).__init__()
        qconfig = model.qconfig if hasattr(model, "qconfig") else None
        self.quant = QuantStub(qconfig)
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, *inputs: torch.Tensor):
        inputs = (self.quant(x) for x in inputs)
        outputs = self.model(*inputs)
        return tuple(self.dequant(x) for x in outputs)


def _quantize_dynamic_torch(model: Module):
    layer_types = {
        type(layer)
        for layer in model.children()
        if len(list(layer.parameters())) > 0
    }
    return torch.quantization.quantize_dynamic(
        model=model, qconfig_spec=layer_types, dtype=torch.qint8
    )


def _quantize_dynamic_torch_fx(
    model: GraphModule,
    input_data: List[Tuple[torch.Tensor, ...]],
):
    qconfig_dict = {"": default_dynamic_qconfig}

    if check_module_version(torch, min_version="1.13.0"):
        model_prepared = prepare_fx(
            model, qconfig_dict, example_inputs=input_data[0]
        )
    else:
        model_prepared = prepare_fx(model, qconfig_dict)
    return convert_fx(model_prepared)


def _quantize_static_torch(
    model: Module,
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
    model: GraphModule,
    input_data: List[Tuple[torch.Tensor, ...]],
    backend: str,
):
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    if check_module_version(torch, min_version="1.13.0"):
        model_prepared = prepare_fx(
            model, qconfig_dict, example_inputs=input_data[0]
        )
    else:
        model_prepared = prepare_fx(model, qconfig_dict)
    with torch.no_grad():
        for tensors in input_data:
            _ = model_prepared(*tensors)
    return convert_fx(model_prepared)


def _quantize_static(
    model: Union[Module, GraphModule],
    input_data: List[Tuple[torch.Tensor, ...]],
    device: Device,
):
    assert (
        device is not Device.GPU
    ), "Quantization for torch is only available on CPU"

    backend = (
        "fbgemm"
        if "fbgemm" in torch.backends.quantized.supported_engines
        else "qnnpack"
    )

    torch.backends.quantized.engine = backend

    if isinstance(model, GraphModule):
        return _quantize_static_torch_fx(model, input_data, backend)
    else:
        return _quantize_static_torch(model, input_data, backend)


def _quantize_dynamic(
    model: Union[Module, GraphModule],
    input_data: List[Tuple[torch.Tensor, ...]],
    device: Device,
):
    assert (
        device is not Device.GPU
    ), "Quantization for torch is only available on CPU"

    backend = (
        "fbgemm"
        if "fbgemm" in torch.backends.quantized.supported_engines
        else "qnnpack"
    )

    torch.backends.quantized.engine = backend

    if isinstance(model, GraphModule):
        return _quantize_dynamic_torch_fx(model, input_data)
    else:
        return _quantize_dynamic_torch(model)


def _half_precision(model: Module):
    return model.half()


class PytorchQuantizer(Quantizer):
    def execute(
        self,
        model: Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data_torch: List[Tuple[torch.Tensor, ...]],
    ):
        model = copy.deepcopy(model).eval()

        try:
            model = symbolic_trace(model)
        except Exception:
            self.logger.warning("Unable to trace model with torch.fx")

        if quantization_type is QuantizationType.HALF:
            input_tfms.append(HalfPrecisionTransformation())
            self.quantized_model = _half_precision(model)
        elif quantization_type is QuantizationType.STATIC:
            self.quantized_model, _ = (
                _quantize_static(model, input_data_torch, self.device),
                input_tfms,
            )
        elif quantization_type is QuantizationType.DYNAMIC:
            self.quantized_model, _ = (
                _quantize_dynamic(model, input_data_torch, self.device),
                input_tfms,
            )
        else:
            raise NotImplementedError(
                f"No quantization implemented for quantization "
                f"type {quantization_type}"
            )
