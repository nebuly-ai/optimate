import logging
from typing import List, Dict, Type

from nebullvm.base import ModelCompiler
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.operations.inference_learners.builders import (
    DeepSparseBuildInferenceLearner,
    OpenVINOBuildInferenceLearner,
    PytorchBuildInferenceLearner, ONNXBuildInferenceLearner,
)
from nebullvm.operations.measures.measures import PrecisionMeasure
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.deepsparse import DeepSparseCompiler
from nebullvm.operations.optimizations.compilers.onnx import ONNXCompiler
from nebullvm.operations.optimizations.compilers.openvino import OpenVINOCompiler
from nebullvm.operations.optimizations.compilers.pytorch import (
    PytorchBackendCompiler,
)
from nebullvm.utils.compilers import select_compilers_from_hardware_torch, select_compilers_from_hardware_onnx

logger = logging.getLogger("nebullvm_logger")


class PytorchOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.compiler_ops = {}
        self.build_inference_learner_ops = {}
        self.validity_check_op = PrecisionMeasure()

    def _load_compilers(
        self, ignore_compilers: List[ModelCompiler]
    ):
        compilers = select_compilers_from_hardware_torch(self.device)
        self.compiler_ops = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler]()
            for compiler in compilers
            if compiler not in ignore_compilers and compiler in COMPILER_TO_OPTIMIZER_MAP
        }
        self.build_inference_learner_ops = {
            compiler: COMPILER_TO_INFERENCE_LEARNER_MAP[compiler]()
            for compiler in compilers
            if compiler not in ignore_compilers and compiler in COMPILER_TO_OPTIMIZER_MAP
        }

    def execute(
        self,
        model,
        input_data,
        optimization_time,
        metric_drop_ths,
        metric,
        model_params,
        model_outputs,
        ignore_compilers
    ):
        self._load_compilers(ignore_compilers=ignore_compilers)
        self.optimize(
            model,
            input_data,
            optimization_time,
            metric_drop_ths,
            metric,
            model_params,
            model_outputs,
        )


class TensorflowOptimizer(Optimizer):
    def _get_compilers(self) -> List[Compiler]:
        pass

    def execute(self, **kwargs):
        pass


class ONNXOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.compiler_ops = {}
        self.build_inference_learner_ops = {}
        self.validity_check_op = PrecisionMeasure()

    def _load_compilers(
            self, ignore_compilers: List[ModelCompiler]
    ):
        compilers = select_compilers_from_hardware_onnx(self.device)
        self.compiler_ops = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler]()
            for compiler in compilers
            if compiler not in ignore_compilers and compiler in COMPILER_TO_OPTIMIZER_MAP
        }
        self.build_inference_learner_ops = {
            compiler: COMPILER_TO_INFERENCE_LEARNER_MAP[compiler]()
            for compiler in compilers
            if compiler not in ignore_compilers and compiler in COMPILER_TO_OPTIMIZER_MAP
        }

    def execute(
        self,
        model,
        input_data,
        optimization_time,
        metric_drop_ths,
        metric,
        model_params,
        model_outputs,
        ignore_compilers,
    ):
        self._load_compilers(ignore_compilers=ignore_compilers)
        self.optimize(
            model,
            input_data,
            optimization_time,
            metric_drop_ths,
            metric,
            model_params,
            model_outputs,
        )


COMPILER_TO_OPTIMIZER_MAP: Dict[ModelCompiler, Type[Compiler]] = {
    ModelCompiler.TORCHSCRIPT: PytorchBackendCompiler,
    ModelCompiler.DEEPSPARSE: DeepSparseCompiler,
    ModelCompiler.ONNX_RUNTIME: ONNXCompiler,
    ModelCompiler.OPENVINO: OpenVINOCompiler,
}

COMPILER_TO_INFERENCE_LEARNER_MAP: Dict[ModelCompiler, Type[BuildInferenceLearner]] = {
    ModelCompiler.TORCHSCRIPT: PytorchBuildInferenceLearner,
    ModelCompiler.DEEPSPARSE: DeepSparseBuildInferenceLearner,
    ModelCompiler.ONNX_RUNTIME: ONNXBuildInferenceLearner,
    ModelCompiler.OPENVINO: OpenVINOBuildInferenceLearner,
}
