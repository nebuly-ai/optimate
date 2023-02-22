import copy
from typing import List, Tuple, Union

from loguru import logger

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
    ScriptModule,
)
from nebullvm.tools.base import Device, QuantizationType, DeviceType
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

    additional_arguments = {}
    if check_module_version(torch, min_version="1.13.0"):
        additional_arguments["example_inputs"] = input_data[0]

    model_prepared = prepare_fx(model, qconfig_dict, **additional_arguments)
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
    additional_arguments = {}
    if check_module_version(torch, min_version="1.13.0"):
        additional_arguments["example_inputs"] = input_data[0]

    model_prepared = prepare_fx(model, qconfig_dict, **additional_arguments)
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
        device is not DeviceType.GPU
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
        device is not DeviceType.GPU
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


def quantize_pytorch(
    model: Module,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    input_data_torch: List[Tuple[torch.Tensor, ...]],
    device: Device,
) -> Union[torch.nn.Module, ScriptModule, GraphModule]:
    model = copy.deepcopy(model).eval()

    try:
        model = symbolic_trace(model)
    except Exception:
        logger.warning("Unable to trace model with torch.fx")

    if quantization_type is QuantizationType.HALF:
        input_tfms.append(HalfPrecisionTransformation())
        quantized_model = _half_precision(model)
    elif quantization_type is QuantizationType.STATIC:
        quantized_model = _quantize_static(model, input_data_torch, device)
    elif quantization_type is QuantizationType.DYNAMIC:
        quantized_model = _quantize_dynamic(model, input_data_torch, device)
    else:
        raise NotImplementedError(
            f"No quantization implemented for quantization "
            f"type {quantization_type}"
        )

    return quantized_model
