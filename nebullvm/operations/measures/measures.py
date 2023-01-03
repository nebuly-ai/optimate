from typing import List, Tuple, Any, Callable, Dict, Optional

import numpy as np

from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.operations.measures.base import Measure
from nebullvm.operations.measures.utils import (
    compute_torch_latency,
    compute_tf_latency,
    compute_onnx_latency,
    compute_relative_difference,
)
from nebullvm.tools.base import DeepLearningFramework
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import run_onnx_model
from nebullvm.tools.pytorch import run_torch_model
from nebullvm.tools.tf import run_tf_model

COMPUTE_OUTPUT_FRAMEWORK: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: run_torch_model,
    DeepLearningFramework.TENSORFLOW: run_tf_model,
    DeepLearningFramework.NUMPY: run_onnx_model,
}

COMPUTE_LATENCY_FRAMEWORK: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: compute_torch_latency,
    DeepLearningFramework.TENSORFLOW: compute_tf_latency,
    DeepLearningFramework.NUMPY: compute_onnx_latency,
}


class MetricDropMeasure(Measure):
    def __init__(self):
        super().__init__()
        self.valid = None

    def execute(
        self,
        optimized_learner: BaseInferenceLearner,
        input_data: List[Tuple[Any, ...]],
        base_outputs_list: List[Tuple[Any, ...]],
        perf_loss_ths: float,
        metric_func: Callable = None,
        ys: List = None,
        aggregation_func: Callable = np.mean,
    ):
        metric_func = metric_func or compute_relative_difference
        relative_differences = []
        if ys is None:
            ys = [None] * len(input_data)

        assert len(input_data) == len(base_outputs_list) == len(ys), (
            "INTERNAL ASSERT FAILED: error during computation of precision "
            "of the optimized model, got wrong dimensions of the data. "
        )

        for inputs, base_outputs, y in zip(input_data, base_outputs_list, ys):
            opt_outputs = optimized_learner(*inputs)
            relative_difference = max(
                metric_func(base_output, opt_output, y)
                for base_output, opt_output in zip(base_outputs, opt_outputs)
            )
            relative_differences.append(relative_difference)
        relative_difference = aggregation_func(relative_differences)
        self.valid = relative_difference <= perf_loss_ths
        self.measure_result = relative_difference

    def get_result(self) -> Tuple:
        return self.valid, self.measure_result


class LatencyOriginalModelMeasure(Measure):
    def __init__(self):
        super().__init__()
        self.outputs = None

    def execute(
        self,
        model: Any,
        input_data: DataManager,
        dl_framework: DeepLearningFramework,
    ):
        self.logger.info("Benchmark performance of original model")

        self.outputs = [
            tuple(
                COMPUTE_OUTPUT_FRAMEWORK[dl_framework](
                    model, tuple(input_tensors[0]), self.device
                )
            )
            for input_tensors in input_data
        ]

        inputs = input_data.get_list(QUANTIZATION_DATA_NUM)
        self.measure_result, _ = COMPUTE_LATENCY_FRAMEWORK[dl_framework](
            inputs, model, self.device
        )
        self.logger.info(
            f"Original model latency: {self.measure_result} sec/iter"
        )

    def get_result(self) -> Optional[Tuple]:
        if self.outputs is not None and self.measure_result is not None:
            return self.outputs, self.measure_result
        else:
            return None
