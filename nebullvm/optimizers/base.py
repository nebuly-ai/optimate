from abc import abstractmethod, ABC
from logging import Logger

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.inference_learners.base import BaseInferenceLearner


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
