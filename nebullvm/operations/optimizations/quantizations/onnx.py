from pathlib import Path
from typing import Union, Iterable, Tuple, List

import cpuinfo
import numpy as np

from nebullvm.operations.optimizations.quantizations.base import Quantizer
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
from nebullvm.tools.base import QuantizationType, Device
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


def _quantize_dynamic(model_path: str):
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
):
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
):
    model_path = Path(model_path)
    model_quant = model_path.parent.parent / "fp16"
    model_quant.mkdir(parents=True)
    model_quant = model_quant / (model_path.stem + "_fp16.onnx")
    new_onnx_model = convert_float_to_float16_model_path(str(model_path))
    input_tfms.append(HalfPrecisionTransformation())
    onnx.save(new_onnx_model, str(model_quant))
    return str(model_quant)


class ONNXQuantizer(Quantizer):
    def execute(
        self,
        model_path: str,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data: List[Tuple[np.ndarray, ...]],
    ):
        use_gpu = self.device is Device.GPU

        if quantization_type is QuantizationType.STATIC:
            self.quantized_model = _quantize_static(
                model_path, input_data, use_gpu
            )
        elif quantization_type is QuantizationType.DYNAMIC:
            self.quantized_model = _quantize_dynamic(model_path)
        elif quantization_type is QuantizationType.HALF:
            self.quantized_model = _convert_to_half_precision(
                model_path, input_tfms
            )
        else:
            raise ValueError(
                f"Quantization type {quantization_type} is not supported"
            )
