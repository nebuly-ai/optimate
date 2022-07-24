import warnings
from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, List, Any, Callable, Tuple, Optional

import cpuinfo
import numpy as np
import tensorflow as tf
import torch.nn

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    QuantizationType,
    ModelCompiler,
)
from nebullvm.compressors.base import BaseCompressor
from nebullvm.compressors.intel import TorchIntelPruningCompressor
from nebullvm.inference_learners import (
    BaseInferenceLearner,
    PytorchBaseInferenceLearner,
)
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    BaseOptimizer,
    ApacheTVMOptimizer,
    COMPILER_TO_OPTIMIZER_MAP,
)
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
from nebullvm.optimizers.tensorflow import TensorflowBackendOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.compilers import (
    tvm_is_available,
    select_compilers_from_hardware_onnx,
)
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR


class Step(ABC):
    def __init__(self, logger: Logger = None):
        self._logger = logger

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict:
        raise NotImplementedError()


class CompressorStep(Step, ABC):
    def __init__(self, config_file: str = None, logger: Logger = None):
        super().__init__(logger)
        self._config_file = config_file

    def run(
        self,
        model: Any = None,
        input_data: DataManager = None,
        metric_drop_ths: float = None,
        metric: Callable = None,
        **kwargs,
    ) -> Dict:
        compressor_dict = self._get_compressors()
        models = {}
        train_input_data, eval_input_data = input_data.split(0.8)
        for technique, compressor in compressor_dict.items():
            compressed_model, ths = compressor.compress(
                model,
                train_input_data,
                eval_input_data,
                metric_drop_ths,
                metric,
            )
            models[technique] = (compressed_model, ths)
        return {
            "models": models,
            "input_data": eval_input_data,
            "metric": metric,
            **kwargs,
        }

    @abstractmethod
    def _get_compressors(self) -> Dict[str, BaseCompressor]:
        raise NotImplementedError()


class TorchCompressorStep(CompressorStep):
    def _get_compressors(self) -> Dict[str, BaseCompressor]:
        compressors = {}
        if "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower():
            compressors["intel_pruning"] = TorchIntelPruningCompressor(
                config_file=self._config_file
            )
        return compressors


class NoCompressionStep(Step):
    def run(self, model: Any, **kwargs) -> Dict:
        return {"models": {"": model}, **kwargs}


class OptimizerStep(Step, ABC):
    def run(
        self,
        models: Dict[str, Tuple[Any, Optional[float]]] = None,
        output_library: DeepLearningFramework = None,
        model_params: ModelParams = None,
        input_tfms: MultiStageTransformation = None,
        metric: Optional[Callable] = None,
        input_data: Optional[DataManager] = None,
        ignore_compilers: List[ModelCompiler] = None,
        **kwargs,
    ) -> Dict:

        optimizers = self._get_optimizers(ignore_compilers)
        optimized_models = []

        for prev_tech, (model, metric_drop_ths) in models.items():
            if model is None:
                continue
            if metric_drop_ths is not None:
                q_types = [
                    None,
                    QuantizationType.DYNAMIC,
                    QuantizationType.HALF,
                ]
                if input_data is not None:
                    q_types.append(QuantizationType.STATIC)
            else:
                q_types = [None]
            for compiler, optimizer in optimizers.items():
                for q_type in q_types:
                    try:
                        optimized_model = self._run_optimizer(
                            optimizer,
                            model,
                            output_library,
                            model_params,
                            input_tfms.copy(),
                            metric_drop_ths,
                            q_type,
                            metric,
                            input_data,
                        )
                        if optimized_model is not None:
                            latency = compute_optimized_running_time(
                                optimized_model
                            )
                        else:
                            latency = np.inf
                        optimized_models.append((optimized_model, latency))
                        if compiler not in ignore_compilers:
                            ignore_compilers.append(compiler)
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            perf_loss_ths=metric_drop_ths,
                            latency=latency,
                            compression=prev_tech,
                        )
                    except Exception as ex:
                        warnings.warn(
                            f"Compilation failed with {output_library.value} "
                            f"interface of {compiler}. Got error {ex}. "
                            f"If possible the compilation will be re-scheduled"
                            f" with another interface. Please consult the "
                            f"documentation for further info or open an issue "
                            f"on GitHub for receiving assistance."
                        )
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            perf_loss_ths=metric_drop_ths,
                            latency=None,
                            compression=prev_tech,
                        )

        return {
            "optimized_models": optimized_models,
            "ignore_compilers": ignore_compilers,
        }

    @abstractmethod
    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        perf_loss_ths: float = None,
        quantization_type: QuantizationType = None,
        perf_metric: Callable = None,
        input_data: DataManager = None,
    ) -> BaseInferenceLearner:
        raise NotImplementedError()

    @abstractmethod
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        raise NotImplementedError()


class TorchOptimizerStep(OptimizerStep):
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {
            ModelCompiler.TORCHSCRIPT: PytorchBackendOptimizer(
                logger=self._logger
            ),
        }
        if (
            tvm_is_available()
            and ModelCompiler.APACHE_TVM not in ignore_compilers
        ):
            optimizers[ModelCompiler.APACHE_TVM] = ApacheTVMOptimizer(
                logger=self._logger
            )
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        perf_loss_ths: float = None,
        quantization_type: QuantizationType = None,
        perf_metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_torch"):
            optimized_model = optimizer.optimize_from_torch(
                torch_model=model,
                model_params=model_params,
                perf_loss_ths=perf_loss_ths
                if quantization_type is not None
                else None,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                perf_loss_ths=perf_loss_ths
                if quantization_type is not None
                else None,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        return optimized_model


class TFOptimizerStep(OptimizerStep):
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {
            ModelCompiler.TFLITE: TensorflowBackendOptimizer(
                logger=self._logger
            )
        }
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        perf_loss_ths: float = None,
        quantization_type: QuantizationType = None,
        perf_metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_tf"):
            optimized_model = optimizer.optimize_from_tf(
                torch_model=model,
                model_params=model_params,
                perf_loss_ths=perf_loss_ths
                if quantization_type is not None
                else None,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                perf_loss_ths=perf_loss_ths
                if quantization_type is not None
                else None,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        return optimized_model


class OnnxOptimizerStep(OptimizerStep):
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        compilers = select_compilers_from_hardware_onnx()
        optimizers = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler](self._logger)
            for compiler in compilers
            if compiler not in ignore_compilers
        }
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        perf_loss_ths: float = None,
        quantization_type: QuantizationType = None,
        perf_metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        optimized_model = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            perf_loss_ths=perf_loss_ths
            if quantization_type is not None
            else None,
            quantization_type=quantization_type,
            input_tfms=input_tfms,
            input_data=input_data,
        )
        return optimized_model


class Pipeline(Step):
    def __init__(self, steps: List[Step], logger: Logger = None):
        super().__init__(logger)
        self._steps = steps

    def run(self, **kwargs) -> Dict:
        for step in self._steps:
            kwargs = step.run(**kwargs)
        return kwargs


def _get_compressor_step(
    model: Any,
    optimization_time: str,
    config_file: Optional[str],
    metric_drop_ths: Optional[float],
    metric: Optional[Callable],
    logger: Optional[Logger],
) -> Step:
    if optimization_time == "constrained":
        return NoCompressionStep(logger=logger)
    if metric_drop_ths is None or metric is None:
        return NoCompressionStep(logger=logger)
    elif isinstance(model, torch.nn.Module):
        return TorchCompressorStep(config_file=config_file, logger=logger)
    else:  # default is NoCompression
        return NoCompressionStep(logger=logger)


def _get_optimizer_step(
    model: Any,
    logger: Optional[Logger],
) -> Step:
    if isinstance(model, torch.nn.Module):
        return TorchOptimizerStep(logger=logger)
    elif isinstance(model, tf.Module):
        return TFOptimizerStep(logger=logger)
    else:
        return OnnxOptimizerStep(logger=logger)


def build_pipeline_from_model(
    model: Any,
    optimization_time: str,
    metric_drop_ths: Optional[float],
    metric: Optional[Callable],
    config_file: Optional[str],
    logger: Logger = None,
) -> Pipeline:
    compressor_step = _get_compressor_step(
        model, optimization_time, config_file, metric_drop_ths, metric, logger
    )
    optimizer_step = _get_optimizer_step(model, logger)
    pipeline = Pipeline(logger=logger, steps=[compressor_step, optimizer_step])
    return pipeline
