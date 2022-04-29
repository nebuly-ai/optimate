from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Union, Tuple, Dict, List
import warnings

from nebullvm.transformations.base import MultiStageTransformation


class BaseQuantizer(ABC):
    def __init__(self, tolerated_error: float = 1e-5, logger: Logger = None):
        self.tolerated_error = tolerated_error
        self.logger = logger

    @abstractmethod
    def _quantize(
        self,
        model: Any,
        input_data: List[Tuple],
        input_tfms: MultiStageTransformation,
        **kwargs,
    ) -> Tuple[Any, Dict, MultiStageTransformation]:
        raise NotImplementedError()

    @abstractmethod
    def _read_and_check_model(
        self, model_path: Union[str, Path], **kwargs
    ) -> Tuple[Any, Dict]:
        raise NotImplementedError()

    @abstractmethod
    def _check_and_save_model(
        self, quantized_model: Any, **kwargs
    ) -> Union[str, Path]:
        raise NotImplementedError()

    @abstractmethod
    def _run_model(
        self,
        model: Any,
        input_data: List[Tuple],
        input_tfms: MultiStageTransformation,
    ) -> List[Tuple]:
        """Run the model and get predictions as tuple"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _compare_outputs(out1: Any, out2: Any):
        raise NotImplementedError()

    def _check_model_performance(
        self,
        quantized_model: Any,
        original_model: Any,
        input_data: List[Tuple],
        input_tfms: MultiStageTransformation,
        input_tfms_after_quant: MultiStageTransformation,
    ):
        original_outputs = self._run_model(
            original_model, input_data, input_tfms
        )
        quantized_outputs = self._run_model(
            quantized_model, input_data, input_tfms_after_quant
        )
        error = max(
            self._compare_outputs(out1, out2)
            for outputs_1, outputs_2 in zip(
                original_outputs, quantized_outputs
            )
            for out1, out2 in zip(outputs_1, outputs_2)
        )
        if error > self.tolerated_error:
            message = (
                f"Quantization caused an excessive degradation of the "
                f"prediction. Maximum tolerated error: "
                f"{self.tolerated_error}. "
                f"Detected error: {error}. The quantizer "
                f"{self.__class__.__name__} will be skipped."
            )
            if self.logger is None:
                warnings.warn(message)
            else:
                self.logger.warning(message)
            return False
        return True

    def __call__(
        self,
        model_path: Union[str, Path],
        input_data: List[Tuple],
        input_tfms: MultiStageTransformation,
        **kwargs,
    ) -> Tuple[Union[str, Path], MultiStageTransformation]:
        model, model_kwargs = self._read_and_check_model(model_path, **kwargs)
        new_input_tfms = input_tfms.copy()
        quantized_model, quantized_kwargs, new_input_tfms = self._quantize(
            model,
            input_data=input_data,
            input_tfms=new_input_tfms,
            **model_kwargs,
        )
        if self._check_model_performance(
            quantized_model, model, input_data, input_tfms, new_input_tfms
        ):
            return (
                self._check_and_save_model(
                    quantized_model, **quantized_kwargs
                ),
                new_input_tfms,
            )
        else:
            return "", input_tfms
