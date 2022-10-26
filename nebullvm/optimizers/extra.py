import logging
from logging import Logger
from typing import Dict, List, Optional, Callable, Any

from onnxruntime.transformers.optimizer import MODEL_TYPES

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.inference_learners.onnx import (
    ONNXInferenceLearner,
    ONNX_INFERENCE_LEARNERS,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.utils import check_precision
from nebullvm.optional_modules.onnxruntime import optimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
)


class HuggingFaceOptimizer(BaseOptimizer):
    def __init__(
        self,
        hugging_face_params: Dict,
        metric_drop_ths: float = None,
        metric: Callable = None,
        logger: Logger = None,
    ):
        super(HuggingFaceOptimizer, self).__init__(logger)
        self.hf_params = hugging_face_params
        self.perf_loss_ths = metric_drop_ths
        self.perf_metric = metric
        self.q_type = QuantizationType.HALF

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[ONNXInferenceLearner]:
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        optimized_model = optimizer.optimize_model(model, **self.hf_params)
        if quantization_type is not None:
            if quantization_type is not QuantizationType.HALF:
                return None
            optimized_model.convert_float_to_float16()
            new_onnx_model = model.replace(".onnx", "_fp16.onnx")
        else:
            new_onnx_model = model.replace(".onnx", "_opt.onnx")
        optimized_model.save_model_to_file(new_onnx_model)

        learner = ONNX_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            onnx_path=new_onnx_model,
            input_names=get_input_names(new_onnx_model),
            output_names=get_output_names(new_onnx_model),
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )

        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )
        is_valid = check_precision(
            learner,
            test_input_data,
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
                    "The model optimized with huggingface gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner

    @staticmethod
    def get_accepted_types() -> List[str]:
        return list(MODEL_TYPES.keys())
