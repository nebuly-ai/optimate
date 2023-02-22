from pathlib import Path
from typing import Union, Iterable, Tuple, List

import cpuinfo
import numpy as np

from nebullvm.optional_modules.onnx import (
    onnx,
    convert_float_to_float16_model_path,
)
from nebullvm.optional_modules.onnxruntime import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from nebullvm.optional_modules.torch import DataLoader
from nebullvm.tools.base import QuantizationType, Device, DeviceType
from nebullvm.tools.onnx import get_input_names
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    HalfPrecisionTransformation,
)


class _IterableCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        iterable_dataset: Union[Iterable[Tuple], List[Tuple]],
        input_names: List[str],
    ):
        self.iterable_dataset = iter(
            [
                {
                    input_name: value
                    for inputs in iterable_dataset
                    for input_name, value in zip(input_names, inputs)
                }
            ]
        )

    def get_next(self) -> dict:
        return next(self.iterable_dataset, None)

    @classmethod
    def from_dataloader(
        cls, dl: DataLoader, input_names: List[str], contains_y: bool = True
    ):
        iterable_ds = iter(
            inputs[:-1] if contains_y else inputs for inputs in dl
        )
        return cls(iterable_ds, input_names)


def _quantize_dynamic(model_path: str) -> str:
    model_path = Path(model_path)
    model_quant = model_path.parent.parent / "int8_dynamic"
    model_quant.mkdir(parents=True, exist_ok=True)
    model_quant = model_quant / (model_path.stem + ".quant.onnx")
    quantize_dynamic(
        model_path,
        model_quant,
        weight_type=QuantType.QUInt8,
        optimize_model=False,
    )
    return str(model_quant)


def _get_quantization_type_for_static(use_gpu) -> Tuple[QuantType, QuantType]:
    """Returns the quantization types for activations and weights,
    depending on the underlying hardware
    """
    arch = cpuinfo.get_cpu_info()["arch"].lower()
    if use_gpu:
        activation_type = weight_type = QuantType.QInt8
    elif "x86" in arch:
        cpu_raw_data = cpuinfo.get_cpu_info()["brand_raw"].lower()
        if "intel" in cpu_raw_data and "xeon" in cpu_raw_data:
            activation_type = QuantType.QUInt8
            weight_type = QuantType.QInt8
        else:
            activation_type = weight_type = QuantType.QUInt8
    else:
        activation_type = QuantType.QUInt8
        weight_type = QuantType.QUInt8
    return activation_type, weight_type


def _quantize_static(
    model_path: str, input_data: List[Tuple[np.ndarray, ...]], use_gpu: bool
) -> str:
    model_path = Path(model_path)
    model_quant = model_path.parent.parent / "int8_static"
    model_quant.mkdir(parents=True, exist_ok=True)
    model_quant = model_quant / (model_path.stem + ".quant.onnx")
    inputs = input_data
    input_names = get_input_names(str(model_path))
    cdr = _IterableCalibrationDataReader(
        input_names=input_names, iterable_dataset=inputs
    )
    activation_type, weight_type = _get_quantization_type_for_static(use_gpu)
    quantize_static(
        model_path,
        Path(model_quant),
        cdr,
        activation_type=activation_type,
        weight_type=weight_type,
        optimize_model=False,
    )
    return str(model_quant)


def _convert_to_half_precision(
    model_path: str, input_tfms: MultiStageTransformation
) -> str:
    model_path = Path(model_path)
    model_quant = model_path.parent.parent / "fp16"
    model_quant.mkdir(parents=True)
    model_quant = model_quant / (model_path.stem + "_fp16.onnx")
    new_onnx_model = convert_float_to_float16_model_path(str(model_path))
    input_tfms.append(HalfPrecisionTransformation())
    try:
        onnx.save(new_onnx_model, str(model_quant))
    except ValueError:
        # Model larger than 2GB must be saved as external data
        onnx.save(
            new_onnx_model,
            str(model_quant),
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            convert_attribute=True,
        )
    return str(model_quant)


def quantize_onnx(
    model_path: str,
    input_data: List[Tuple[np.ndarray, ...]],
    quantization_type: QuantizationType,
    device: Device,
    input_tfms: MultiStageTransformation,
) -> str:
    if quantization_type == QuantizationType.DYNAMIC:
        return _quantize_dynamic(model_path)
    elif quantization_type == QuantizationType.STATIC:
        return _quantize_static(
            model_path, input_data, device.type is DeviceType.GPU
        )
    elif quantization_type == QuantizationType.HALF:
        return _convert_to_half_precision(model_path, input_tfms)
    else:
        raise ValueError(
            f"Quantization type {quantization_type} not supported"
        )
