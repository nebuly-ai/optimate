import warnings
from functools import partial
from logging import Logger
from typing import Dict, Type, Tuple, Callable

import cpuinfo
import numpy as np
from joblib import Parallel, delayed
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
    def __init__(self, logger: Logger = None, n_jobs: int = 1):
        super().__init__(logger)
        self.compilers = select_compilers_from_hardware()
        self.n_jobs = n_jobs

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> BaseInferenceLearner:
        optimization_func = partial(
            _optimize_with_compiler,
            logger=self.logger,
            onnx_model=onnx_model,
            output_library=output_library,
            model_params=model_params,
        )
        optimized_models = Parallel(n_jobs=self.n_jobs)(
            delayed(optimization_func)(compiler) for compiler in self.compilers
        )
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]

    def optimize_on_custom_metric(
        self,
        metric_func: Callable,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        return_all: bool = False,
        n_jobs: int = None,
    ):
        optimization_func = partial(
            _optimize_with_compiler,
            metric_func=metric_func,
            logger=self.logger,
            onnx_model=onnx_model,
            output_library=output_library,
            model_params=model_params,
        )
        optimized_models = Parallel(n_jobs=n_jobs or self.n_jobs)(
            delayed(optimization_func)(compiler) for compiler in self.compilers
        )
        if return_all:
            return optimized_models
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]
