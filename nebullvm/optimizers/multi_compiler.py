import warnings
from logging import Logger
from typing import Dict, Type, Tuple, Callable, List

import cpuinfo
import numpy as np
import torch


from nebullvm.base import ModelCompiler, DeepLearningFramework, ModelParams
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    BaseOptimizer,
    TensorRTOptimizer,
    ApacheTVMOptimizer,
    OpenVinoOptimizer,
)

COMPILER_TO_OPTIMIZER_MAP: Dict[ModelCompiler, Type[BaseOptimizer]] = {
    ModelCompiler.APACHE_TVM: ApacheTVMOptimizer,
    ModelCompiler.OPENVINO: OpenVinoOptimizer,
    ModelCompiler.TENSOR_RT: TensorRTOptimizer,
}


def select_compilers_from_hardware():
    compilers = [ModelCompiler.APACHE_TVM]
    if torch.cuda.is_available():
        compilers.append(ModelCompiler.TENSOR_RT)
    cpu_raw_info = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" in cpu_raw_info:
        compilers.append(ModelCompiler.OPENVINO)
    return compilers


def _optimize_with_compiler(
    compiler: ModelCompiler,
    logger: Logger,
    metric_func: Callable = None,
    **kwargs,
) -> Tuple[BaseInferenceLearner, float]:
    if metric_func is None:
        metric_func = compute_optimized_running_time
    optimizer = COMPILER_TO_OPTIMIZER_MAP[compiler](logger)
    try:
        model_optimized = optimizer.optimize(**kwargs)
        latency = metric_func(model_optimized)
    except Exception as ex:
        warning_msg = (
            f"Compilation failed with {compiler.value}. Got error {ex}."
            f"The compiler will be skipped."
        )
        if logger is None:
            warnings.warn(warning_msg)
        else:
            logger.warning(warning_msg)
        latency = np.inf
        model_optimized = None
    return model_optimized, latency


class MultiCompilerOptimizer(BaseOptimizer):
    def __init__(
        self,
        logger: Logger = None,
        ignore_compilers: List = None,
    ):
        super().__init__(logger)
        self.compilers = [
            compiler
            for compiler in select_compilers_from_hardware()
            if compiler not in (ignore_compilers or [])
        ]

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> BaseInferenceLearner:
        optimized_models = [
            _optimize_with_compiler(
                compiler,
                logger=self.logger,
                onnx_model=onnx_model,
                output_library=output_library,
                model_params=model_params,
            )
            for compiler in self.compilers
        ]
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]

    def optimize_on_custom_metric(
        self,
        metric_func: Callable,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        return_all: bool = False,
    ):
        optimized_models = [
            _optimize_with_compiler(
                compiler,
                metric_func=metric_func,
                logger=self.logger,
                onnx_model=onnx_model,
                output_library=output_library,
                model_params=model_params,
            )
            for compiler in self.compilers
        ]
        if return_all:
            return optimized_models
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]

    @property
    def usable(self) -> bool:
        return len(self.compilers) > 0
