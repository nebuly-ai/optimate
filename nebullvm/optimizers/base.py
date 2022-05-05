from abc import abstractmethod, ABC
from logging import Logger
from typing import Optional, Callable

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


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
        input_tfms: MultiStageTransformation = None,
        quantization_ths: float = None,
        quantization_type: QuantizationType = None,
        quantization_metric: Callable = None,
        input_data: DataManager = None,
    ) -> Optional[BaseInferenceLearner]:
        raise NotImplementedError
