from abc import abstractmethod, ABC
from logging import Logger

import onnx

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.inference_learners.base import BaseInferenceLearner


def get_input_names(onnx_model: str):
    model = onnx.load(onnx_model)

    input_all = [node.name for node in model.graph.input]
    return input_all


def get_output_names(onnx_model: str):
    model = onnx.load(onnx_model)
    output_all = [node.name for node in model.graph.output]
    return output_all


class BaseOptimizer(ABC):
    """Base class for Optimizers"""

    def __init__(self, logger: Logger = None):
        self.logger = logger

    @abstractmethod
    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> BaseInferenceLearner:
        raise NotImplementedError
