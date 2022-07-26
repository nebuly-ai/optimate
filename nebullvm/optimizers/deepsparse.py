from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Callable

import torch

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.converters import ONNXConverter
from nebullvm.inference_learners.deepsparse import (
    DEEPSPARSE_INFERENCE_LEARNERS,
    DeepSparseInferenceLearner,
)
from nebullvm.optimizers import BaseOptimizer
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
    ) -> Optional[DeepSparseInferenceLearner]:
        if quantization_type is not None:
            return

        with TemporaryDirectory() as tmp_dir:
            converter = ONNXConverter()
            onnx_pruned_path = Path(tmp_dir) / "model_pruned.onnx"
            converter.convert(
                model, model_params, onnx_pruned_path, input_data
            )

            learner = DEEPSPARSE_INFERENCE_LEARNERS[output_library](
                input_tfms=input_tfms,
                network_parameters=model_params,
                onnx_path=onnx_pruned_path,
                input_names=get_input_names(str(onnx_pruned_path)),
                output_names=get_output_names(str(onnx_pruned_path)),
            )
        return learner
