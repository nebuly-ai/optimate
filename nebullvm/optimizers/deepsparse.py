import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Callable, Any

import torch

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.config import QUANTIZATION_DATA_NUM, CONSTRAINED_METRIC_DROP_THS
from nebullvm.converters import ONNXConverter
from nebullvm.inference_learners.deepsparse import (
    DEEPSPARSE_INFERENCE_LEARNERS,
    DeepSparseInferenceLearner,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.utils import check_precision
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
)


class DeepSparseOptimizer(BaseOptimizer):
    def optimize(
        self,
        model: torch.nn.Module,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[DeepSparseInferenceLearner]:
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        if quantization_type is not None:
            return

        with TemporaryDirectory() as tmp_dir:
            converter = ONNXConverter(model_name="model_pruned")
            onnx_pruned_path = Path(tmp_dir)
            converter.convert(
                model, model_params, onnx_pruned_path, input_data
            )
            onnx_pruned_path = str(onnx_pruned_path / "model_pruned.onnx")

            learner = DEEPSPARSE_INFERENCE_LEARNERS[output_library](
                input_tfms=input_tfms,
                network_parameters=model_params,
                onnx_path=onnx_pruned_path,
                input_names=get_input_names(str(onnx_pruned_path)),
                output_names=get_output_names(str(onnx_pruned_path)),
            )

        input_data_torch, ys = input_data.get_list(
            QUANTIZATION_DATA_NUM, with_ys=True
        )

        is_valid = check_precision(
            learner,
            input_data_torch,
            model_outputs,
            metric_drop_ths
            if quantization_type is not None
            else CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )
        if not is_valid:
            if quantization_type is None:
                self._log(
                    "The model optimized with deepsparse gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner
