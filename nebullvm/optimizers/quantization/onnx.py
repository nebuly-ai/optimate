from pathlib import Path
from typing import Union, Iterable, Tuple, List

import cpuinfo
import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

from nebullvm.base import QuantizationType
from nebullvm.config import NO_COMPILER_INSTALLATION
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.transformations.precision_tfms import HalfPrecisionTransformation
from nebullvm.utils.onnx import get_input_names

try:
    from onnxmltools.utils.float16_converter import (
        convert_float_to_float16_model_path,
    )
    from onnxruntime.quantization import (
        QuantType,
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
    )
except ImportError:
    if NO_COMPILER_INSTALLATION:
        QuantType = quantize_static = quantize_dynamic = None
        CalibrationDataReader = object
    else:
        from nebullvm.installers.installers import install_onnxruntime

        install_onnxruntime()
        from onnxruntime.quantization import (
            QuantType,
            quantize_static,
            quantize_dynamic,
            CalibrationDataReader,
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


def _quantize_dynamic(model_path: str):
    model_path = Path(model_path)
    model_quant = model_path.parent / (model_path.stem + ".quant.onnx")
    quantize_dynamic(
        model_path,
        model_quant,
        weight_type=QuantType.QUInt8,
        optimize_model=False,
    )
    return str(model_quant)


def _get_quantization_type_for_static() -> Tuple[QuantType, QuantType]:
    """Returns the quantization types for activations and weights,
    depending on the underlying hardware
    """
    arch = cpuinfo.get_cpu_info()["arch"].lower()
    if torch.cuda.is_available():
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
        weight_type = QuantType.QInt8
    return activation_type, weight_type


def _quantize_static(
    model_path: str, input_data: List[Tuple[np.ndarray, ...]]
):
    model_path = Path(model_path)
    model_quant = model_path.parent / (model_path.stem + ".quant.onnx")
    inputs = input_data
    input_names = get_input_names(str(model_path))
    cdr = _IterableCalibrationDataReader(
        input_names=input_names, iterable_dataset=inputs
    )
    activation_type, weight_type = _get_quantization_type_for_static()
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
):
    model_path = Path(model_path)
    model_quant = model_path.parent / (model_path.stem + "_fp16.onnx")
    new_onnx_model = convert_float_to_float16_model_path(model_path)
    input_tfms.append(HalfPrecisionTransformation())
    onnx.save(new_onnx_model, str(model_quant))
    return str(model_quant), input_tfms


def quantize_onnx(
    model_path: str,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    input_data: List[Tuple[np.ndarray, ...]],
):
    if quantization_type is QuantizationType.STATIC:
        model_path = _quantize_static(model_path, input_data)
    elif quantization_type is QuantizationType.DYNAMIC:
        model_path = _quantize_dynamic(model_path)
    elif quantization_type is QuantizationType.HALF:
        model_path, input_tfms = _convert_to_half_precision(
            model_path, input_tfms
        )

    return model_path, input_tfms
