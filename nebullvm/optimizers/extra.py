from logging import Logger
from typing import Dict, List

from onnxruntime.transformers.optimizer import MODEL_TYPES

from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.inference_learners.onnx import (
    ONNXInferenceLearner,
    ONNX_INFERENCE_LEARNERS,
)
from nebullvm.optimizers import BaseOptimizer
from nebullvm.utils.onnx import get_input_names, get_output_names

try:
    from onnxruntime.transformers import optimizer
except ImportError:
    import warnings

    warnings.warn(
        "No valid onnxruntime installation found. Trying to install it..."
    )
    from nebullvm.installers.installers import install_onnxruntime

    install_onnxruntime()
    from onnxruntime.transformers import optimizer


class HuggingFaceOptimizer(BaseOptimizer):
    def __init__(
        self,
        hugging_face_params: Dict,
        quantization_ths: float = None,
        logger: Logger = None,
    ):
        super(HuggingFaceOptimizer, self).__init__(logger)
        self.hf_params = hugging_face_params
        self.quantization_ths = quantization_ths

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> ONNXInferenceLearner:
        optimized_model = optimizer.optimize_model(
            onnx_model, **self.hf_params
        )
        optimized_model.convert_float_to_float16()
        new_onnx_model = onnx_model.replace(".onnx", "_fp16.onnx")
        optimized_model.save_model_to_file(new_onnx_model)
        learner = ONNX_INFERENCE_LEARNERS[output_library](
            network_parameters=model_params,
            onnx_path=new_onnx_model,
            input_names=get_input_names(onnx_model),
            output_names=get_output_names(onnx_model),
        )
        return learner

    @staticmethod
    def get_accepted_types() -> List[str]:
        return list(MODEL_TYPES.keys())
