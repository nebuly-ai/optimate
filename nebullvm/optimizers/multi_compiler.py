import json
import warnings
from logging import Logger
from pathlib import Path
from typing import Dict, Type, Tuple, Callable, List
import uuid

import numpy as np
from tqdm import tqdm

from nebullvm.base import (
    ModelCompiler,
    DeepLearningFramework,
    ModelParams,
    QuantizationType,
)
from nebullvm.config import NEBULLVM_DEBUG_FILE
from nebullvm.inference_learners.base import BaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    BaseOptimizer,
    COMPILER_TO_OPTIMIZER_MAP,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.compilers import select_compilers_from_hardware_onnx
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR


OPTIMIZER_TO_COMPILER_MAP: Dict[Type[BaseOptimizer], ModelCompiler] = dict(
    zip(COMPILER_TO_OPTIMIZER_MAP.values(), COMPILER_TO_OPTIMIZER_MAP.keys())
)


def _optimize_with_compiler(
    compiler: ModelCompiler,
    logger: Logger,
    metric_func: Callable = None,
    **kwargs,
) -> Tuple[BaseInferenceLearner, float]:
    optimizer = COMPILER_TO_OPTIMIZER_MAP[compiler](logger)
    return _optimize_with_optimizer(optimizer, logger, metric_func, **kwargs)


def _save_info(
    optimizer: BaseOptimizer,
    score: float,
    debug_file: str,
    optimization_params: Dict,
):
    if Path(debug_file).exists():
        with open(debug_file, "r") as f:
            old_dict = json.load(f)
    else:
        old_dict = {}
    optimization_string = optimizer.__class__.__name__
    quantization_string = "_".join(
        [
            str(optimization_params.get(param)) or ""
            for param in ["metric_drop_ths", "quantization_type"]
        ]
    )
    if len(quantization_string) > 1:
        optimization_string += "_" + quantization_string
    old_dict[optimization_string] = f"{score}"
    with open(debug_file, "w") as f:
        json.dump(old_dict, f)


def _optimize_with_optimizer(
    optimizer: BaseOptimizer,
    logger: Logger,
    metric_func: Callable = None,
    debug_file: str = None,
    **kwargs,
) -> Tuple[BaseInferenceLearner, float]:
    if metric_func is None:
        metric_func = compute_optimized_running_time
    try:
        model_optimized = optimizer.optimize(**kwargs)
        latency = metric_func(model_optimized)
        FEEDBACK_COLLECTOR.store_compiler_result(
            OPTIMIZER_TO_COMPILER_MAP[type(optimizer)],
            kwargs["quantization_type"],
            kwargs.get("metric_drop_ths"),
            latency,
        )
    except Exception as ex:
        warning_msg = (
            f"Compilation failed with {optimizer.__class__.__name__}. "
            f"Got error {ex}. The optimizer will be skipped."
        )
        if logger is None:
            warnings.warn(warning_msg)
        else:
            logger.warning(warning_msg)
        latency = np.inf
        model_optimized = None
        FEEDBACK_COLLECTOR.store_compiler_result(
            OPTIMIZER_TO_COMPILER_MAP[type(optimizer)],
            kwargs["quantization_type"],
            kwargs.get("metric_drop_ths"),
            None,
        )
    if debug_file:
        _save_info(optimizer, latency, debug_file, kwargs)
    return model_optimized, latency


class MultiCompilerOptimizer(BaseOptimizer):
    """Run all the optimizers available for the given hardware and select the
    best optimized model in terms of either latency or user defined
    performance.

    Attributes:
        logger (Logger, optional): User defined logger.
        ignore_compilers (List[str], optional): List of compilers that must
            be ignored.
        extra_optimizers (List[BaseOptimizer], optional): List of optimizers
            defined by the user. It usually contains optimizers specific for
            user-defined tasks or optimizers built for the specific model.
            Note that, if given, the optimizers must be already initialized,
            i.e. they could have a different Logger than the one defined in
            `MultiCompilerOptimizer`.
        debug_mode (bool, optional): Boolean flag for activating the debug
            mode. When activated, all the performances of the the different
            containers  will be stored in a json file saved in the working
            directory. Default is False.
    """

    def __init__(
        self,
        logger: Logger = None,
        ignore_compilers: List = None,
        extra_optimizers: List[BaseOptimizer] = None,
        debug_mode: bool = False,
    ):
        super().__init__(logger)
        self.compilers = [
            compiler
            for compiler in select_compilers_from_hardware_onnx()
            if compiler not in (ignore_compilers or [])
        ]
        self.extra_optimizers = extra_optimizers
        self.debug_file = (
            f"{uuid.uuid4()}_{NEBULLVM_DEBUG_FILE}" if debug_mode else None
        )

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> BaseInferenceLearner:
        """Optimize the ONNX model using the available compilers.

        Args:
            model (str): Path to the ONNX model.
            output_library (DeepLearningFramework): Framework of the optimized
                model (either torch on tensorflow).
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            BaseInferenceLearner: Model optimized for inference.
        """
        if metric_drop_ths is not None and quantization_type is None:
            quantization_types = [
                None,
                QuantizationType.DYNAMIC,
                QuantizationType.HALF,
            ]
            if input_data is not None:
                quantization_types.append(QuantizationType.STATIC)
        else:
            quantization_types = [quantization_type]
        optimized_models = [
            _optimize_with_compiler(
                compiler,
                logger=self.logger,
                model=model,
                output_library=output_library,
                model_params=model_params,
                input_tfms=input_tfms.copy()
                if input_tfms is not None
                else None,
                debug_file=self.debug_file,
                metric_drop_ths=metric_drop_ths
                if q_type is not None
                else None,
                quantization_type=q_type,
                metric=metric,
                input_data=input_data,
            )
            for compiler in self.compilers
            for q_type in tqdm(quantization_types)
        ]
        if self.extra_optimizers is not None:
            self._log("Running extra-optimizers...")
            optimized_models += [
                _optimize_with_optimizer(
                    op,
                    logger=self.logger,
                    model=model,
                    output_library=output_library,
                    model_params=model_params,
                    input_tfms=input_tfms.copy()
                    if input_tfms is not None
                    else None,
                    debug_file=self.debug_file,
                    metric_drop_ths=metric_drop_ths
                    if q_type is not None
                    else None,
                    quantization_type=q_type,
                    metric=metric,
                    input_data=input_data,
                )
                for op in self.extra_optimizers
                for q_type in tqdm(quantization_types)
            ]
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]

    def optimize_on_custom_metric(
        self,
        metric_func: Callable,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        return_all: bool = False,
    ):
        """Optimize the ONNX model using the available compilers and give the
        best result sorting by user-defined metric.

        Args:
            metric_func (Callable): function which should be used for sorting
                the compiled models. The metric_func should take as input an
                InferenceLearner and return a numerical value. Note that the
                outputs will be sorted in an ascendant order, i.e. the compiled
                model with the smallest value will be selected.
            model (str): Path to the ONNX model.
            output_library (DeepLearningFramework): Framework of the optimized
                model (either torch on tensorflow).
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            return_all (bool, optional): Boolean flag. If true the method
                returns the tuple (compiled_model, score) for each available
                compiler. Default `False`.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            Union[BaseInferenceLearner, Tuple[BaseInferenceLearner, float]]:
                The method returns just a model optimized for inference if
                `return_all` is `False` or all the compiled models and their
                scores otherwise.
        """
        if metric_drop_ths is not None and quantization_type is None:
            quantization_types = [
                None,
                QuantizationType.DYNAMIC,
                QuantizationType.HALF,
            ]
            if input_data is not None:
                quantization_types.append(QuantizationType.STATIC)
        else:
            quantization_types = [quantization_type]
        optimized_models = [
            _optimize_with_compiler(
                compiler,
                metric_func=metric_func,
                logger=self.logger,
                model=model,
                output_library=output_library,
                model_params=model_params,
                input_tfms=input_tfms.copy()
                if input_tfms is not None
                else None,
                debug_file=self.debug_file,
                perf_loss_ths=metric_drop_ths if q_type is not None else None,
                quantization_type=q_type,
                perf_metric=metric,
                input_data=input_data,
            )
            for compiler in self.compilers
            for q_type in tqdm(quantization_types)
        ]
        if self.extra_optimizers is not None:
            optimized_models += [
                _optimize_with_optimizer(
                    op,
                    logger=self.logger,
                    model=model,
                    output_library=output_library,
                    model_params=model_params,
                    input_tfms=input_tfms.copy()
                    if input_tfms is not None
                    else None,
                    debug_file=self.debug_file,
                    perf_loss_ths=metric_drop_ths
                    if q_type is not None
                    else None,
                    quantization_type=q_type,
                    perf_metric=metric,
                    input_data=input_data,
                )
                for op in self.extra_optimizers
                for q_type in tqdm(quantization_types)
            ]
        if return_all:
            return optimized_models
        optimized_models.sort(key=lambda x: x[1], reverse=False)
        return optimized_models[0][0]

    @property
    def usable(self) -> bool:
        return len(self.compilers) > 0 or (
            self.extra_optimizers is not None
            and len(self.extra_optimizers) > 0
        )
