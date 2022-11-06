from abc import abstractmethod, ABC
from typing import Optional, Callable, Any, Tuple

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


class BaseOptimizer(ABC):
    """Base class for Optimizers"""

    @abstractmethod
    def optimize(
        self,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        device: str,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[Tuple[BaseInferenceLearner, float]]:
        raise NotImplementedError
