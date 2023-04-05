from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any

from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.tools.base import DeepLearningFramework


@dataclass
class HardwareSetup:
    cpu: str
    operating_system: str
    memory_gb: int
    gpu: Optional[str] = None


@dataclass
class OptimizedModel:
    inference_learner: BaseInferenceLearner
    latency_seconds: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    size_mb: float


@dataclass
class OriginalModel:
    model: Any
    latency_seconds: float
    throughput: float
    name: str
    size_mb: float
    framework: DeepLearningFramework


@dataclass
class OptimizeInferenceResult:
    """The result of the OptimizeInferenceOp"""

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_model: Optional[OptimizedModel]

    @property
    def metric_drop(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        return self.optimized_model.metric_drop

    @cached_property
    def latency_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.latency_seconds == 0:
            return -1
        return (
            self.original_model.latency_seconds / self.optimized_model.latency_seconds
        )

    @cached_property
    def throughput_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.original_model.throughput == 0:
            return -1
        return self.optimized_model.throughput / self.original_model.throughput

    @cached_property
    def size_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.size_mb == 0:
            return 1
        return self.original_model.size_mb / self.optimized_model.size_mb
