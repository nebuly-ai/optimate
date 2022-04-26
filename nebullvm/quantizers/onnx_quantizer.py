import time
import warnings
from abc import ABC
from logging import Logger
from pathlib import Path
from typing import Union, Tuple, Dict, Iterable, List, Type, Optional

import cpuinfo
import numpy as np
import onnx
import torch.cuda
from torch.utils.data import DataLoader

from nebullvm.base import QuantizationType, ModelParams
from nebullvm.quantizers.base import BaseQuantizer
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
    create_model_inputs_onnx,
)

try:
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import (
        QuantType,
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
    )
except ImportError:
    from nebullvm.installers.installers import install_onnxruntime

    install_onnxruntime()
    from onnxruntime import InferenceSession
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


class ONNXQuantizer(BaseQuantizer, ABC):
    def __init__(self, tolerated_error: float = 1e-5, logger: Logger = None):
        super().__init__(tolerated_error=tolerated_error, logger=logger)

    def _read_and_check_model(
        self, model_path: Union[str, Path], **kwargs
    ) -> Tuple[str, Dict]:
        model_path = str(model_path)
        onnx.checker.check_model(model_path)
        return model_path, kwargs

    def _check_and_save_model(self, quantized_model: str, **kwargs) -> str:
        # onnx.checker.check_model(quantized_model)
        assert Path(quantized_model).exists()
        return quantized_model

    def _run_model(self, model: str, input_data: List[Tuple]) -> List[Tuple]:
        onnx_model = InferenceSession(model)
        input_names = get_input_names(model)
        output_names = get_output_names(model)
        return [
            onnx_model.run(
                output_names,
                {
                    name: array
                    for name, array in zip(input_names, input_arrays)
                },
            )
            for input_arrays in input_data
        ]

    @staticmethod
    def _compare_outputs(out1: np.ndarray, out2: np.ndarray):
        diff = np.abs(out1 - out2) / (
            np.maximum(np.abs(out1), np.abs(out2)) + 1e-5
        )
        return np.max(diff)


class ONNXDynamicQuantizer(ONNXQuantizer):
    def _quantize(self, model: str, *args, **kwargs) -> Tuple[str, Dict]:
        model_path = Path(model)
        model_quant = model_path.parent / (model_path.stem + ".quant.onnx")
        quantize_dynamic(
            model_path,
            model_quant,
            weight_type=QuantType.QUInt8,
            optimize_model=False,
        )
        return str(model_quant), kwargs


class ONNXStaticQuantizer(ONNXQuantizer):
    def _quantize(self, model: str, *args, **kwargs) -> Tuple[str, Dict]:
        model_path = Path(model)
        model_quant = model_path.parent / (model_path.stem + ".quant.onnx")
        inputs = kwargs.pop("input_data")
        input_names = get_input_names(model)
        cdr = _IterableCalibrationDataReader(
            input_names=input_names, iterable_dataset=inputs
        )
        activation_type, weight_type = self._get_quantization_type()
        quantize_static(
            model_path,
            Path(model_quant),
            cdr,
            activation_type=activation_type,
            weight_type=weight_type,
            optimize_model=False,
        )
        return str(model_quant), kwargs

    @staticmethod
    def _get_quantization_type() -> Tuple[QuantType, QuantType]:
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


ONNX_QUANTIZER_DICT: Dict[QuantizationType, Type[ONNXQuantizer]] = {
    QuantizationType.STATIC: ONNXStaticQuantizer,
    QuantizationType.DYNAMIC: ONNXDynamicQuantizer,
}


class ONNXQuantizerManager:
    def __init__(
        self,
        tolerated_error: float = 1e-2,
        logger: Logger = None,
        steps: int = 100,
    ):
        self.quantizers = {
            q_type: quantizer(tolerated_error=tolerated_error, logger=logger)
            for q_type, quantizer in ONNX_QUANTIZER_DICT.items()
        }
        self.steps = steps
        self.logger = logger

    @staticmethod
    def run_performance(
        onnx_path: str, input_data: List[Tuple[np.ndarray, ...]], steps=100
    ):
        model = InferenceSession(onnx_path)
        input_names = get_input_names(onnx_path)
        output_names = get_output_names(onnx_path)
        times = []
        for _ in range(steps):
            for data in input_data:
                st = time.time()
                _ = model.run(
                    output_names,
                    {name: array for name, array in zip(input_names, data)},
                )
                times.append(time.time() - st)
        return sum(times) / len(times)

    def run(
        self,
        onnx_path: str,
        model_params: ModelParams,
        input_data: List[Tuple[np.ndarray, ...]] = None,
        quantization_type: QuantizationType = None,
    ) -> Optional[str]:
        if input_data is None:
            input_data = [
                tuple(
                    create_model_inputs_onnx(
                        model_params.batch_size, model_params.input_infos
                    )
                )
            ]
        steps = max(round(100 / len(input_data)), 1)
        original_performance = self.run_performance(
            onnx_path, input_data, steps
        )
        if quantization_type is not None:
            q_types = [quantization_type]
        else:
            q_types = list(self.quantizers.keys())
        quantized_paths = (
            self.quantizers[q_type](onnx_path, input_data)
            for q_type in q_types
        )
        quantized_paths = [
            q_path for q_path in quantized_paths if len(q_path) > 0
        ]
        if len(quantized_paths) == 0:
            message = (
                "No Quantization has given a model with the desired tolerated "
                "error. The quantization step will be skipped"
            )
            if self.logger is not None:
                self.logger.warning(message)
            else:
                warnings.warn(message)
            return
        performances = [
            self.run_performance(q_path, input_data, steps)
            for q_path in quantized_paths
        ]
        best_performance = min(performances)
        if original_performance < best_performance:
            message = (
                f"The quantized model is slower than the original one. Got "
                f"original latency: {original_performance} and quantized "
                f"latency {best_performance}. The quantization will be "
                f"skipped."
            )
            if self.logger is not None:
                self.logger.warning(message)
            else:
                warnings.warn(message)
            return
        return quantized_paths[performances.index(best_performance)]
