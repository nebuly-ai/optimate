import abc
from tempfile import TemporaryDirectory
from typing import List, Callable, Union, Tuple, Any

from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.operations.base import Operation
from nebullvm.operations.measures.measures import PrecisionMeasure
from nebullvm.operations.measures.utils import (
    compute_relative_difference,
    compute_optimized_running_time,
)

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import (
    ModelCompiler,
    QuantizationType,
    OptimizationTime,
    ModelParams,
    DeepLearningFramework,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.tools.transformations import MultiStageTransformation


class Optimizer(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimized_models = []
        self.source_dl_framework = None
        self.pipeline_dl_framework = None
        self.compiler_ops = {}
        self.build_inference_learner_ops = {}
        self.validity_check_op = PrecisionMeasure()

    def execute(
        self,
        model: str,
        input_data: DataManager,
        optimization_time: OptimizationTime,
        metric_drop_ths: str,
        metric: Callable,
        model_params: ModelParams,
        model_outputs: List[Tuple[Any, ...]],
        ignore_compilers: List[ModelCompiler],
        source_dl_framework: DeepLearningFramework,
    ):
        self.source_dl_framework = source_dl_framework
        compilers = self._select_compilers_from_hardware()
        (
            self.compiler_ops,
            self.build_inference_learner_ops,
        ) = self._load_compilers(
            ignore_compilers=ignore_compilers,
            compilers=compilers,
        )
        self.optimize(
            model=model,
            input_data=input_data,
            optimization_time=optimization_time,
            metric_drop_ths=metric_drop_ths,
            metric=metric,
            model_params=model_params,
            model_outputs=model_outputs,
            ignore_compilers=ignore_compilers,
        )

    @abc.abstractmethod
    def _select_compilers_from_hardware(self):
        raise NotImplementedError()

    def _load_compilers(
        self,
        ignore_compilers: List[ModelCompiler],
        compilers: List[ModelCompiler],
    ):
        from nebullvm.operations.optimizations.optimizers import (
            COMPILER_TO_OPTIMIZER_MAP,
            COMPILER_TO_INFERENCE_LEARNER_MAP,
        )

        compiler_ops = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler][
                self.pipeline_dl_framework
            ](self.pipeline_dl_framework)
            for compiler in compilers
            if compiler not in ignore_compilers
            and compiler in COMPILER_TO_OPTIMIZER_MAP
        }
        build_inference_learner_ops = {
            compiler: COMPILER_TO_INFERENCE_LEARNER_MAP[compiler][
                self.pipeline_dl_framework
            ](self.pipeline_dl_framework)
            for compiler in compilers
            if compiler not in ignore_compilers
            and compiler in COMPILER_TO_OPTIMIZER_MAP
        }

        return compiler_ops, build_inference_learner_ops

    def optimize(
        self,
        model: Union[torch.nn.Module, tf.Module, str],
        input_data: DataManager,
        optimization_time: OptimizationTime,
        metric_drop_ths: str,
        metric: Callable,
        model_params: ModelParams,
        model_outputs: List[Tuple[Any, ...]],
        ignore_compilers: List[ModelCompiler],
    ):

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

        for compiler, compiler_op, build_inference_learner_op in zip(
            self.compiler_ops.keys(),
            self.compiler_ops.values(),
            self.build_inference_learner_ops.values(),
        ):
            for q_type in q_types:
                input_tfms = MultiStageTransformation([])

                with TemporaryDirectory() as tmp_dir:
                    try:
                        compiler_op.to(self.device).execute(
                            model=model,
                            input_data=input_data,
                            model_params=model_params,
                            metric_drop_ths=metric_drop_ths
                            if q_type is not None
                            else None,
                            quantization_type=q_type,
                            input_tfms=input_tfms,
                            onnx_output_path=tmp_dir,
                        )

                        compiled_model = compiler_op.get_result()
                        if compiled_model is not None:
                            build_inference_learner_op.to(self.device).execute(
                                model=compiled_model,
                                model_orig=compiler_op.model_orig
                                if hasattr(compiler_op, "model_orig")
                                else None,
                                model_params=model_params,
                                input_tfms=input_tfms,
                                dl_framework=self.source_dl_framework,
                            )
                            inference_learner = (
                                build_inference_learner_op.get_result()
                            )

                            if inference_learner is not None:
                                test_input_data, ys = input_data.get_split(
                                    "test"
                                ).get_list(with_ys=True)

                                self.validity_check_op.execute(
                                    inference_learner,
                                    test_input_data,
                                    model_outputs,
                                    metric_drop_ths
                                    if q_type is not None
                                    else CONSTRAINED_METRIC_DROP_THS,
                                    metric_func=metric
                                    if q_type is not None
                                    else compute_relative_difference,
                                    ys=ys,
                                )

                                if self.validity_check_op.valid:
                                    latency = compute_optimized_running_time(
                                        inference_learner, input_data
                                    )
                                    self.logger.info(
                                        f"Optimized model latency: {latency} "
                                        f"sec/iter"
                                    )

                                    if (
                                        compiler not in ignore_compilers
                                        and optimization_time
                                        is OptimizationTime.CONSTRAINED
                                    ):
                                        ignore_compilers.append(compiler)

                                    self.optimized_models.append(
                                        (
                                            inference_learner,
                                            latency,
                                            self.validity_check_op.measure_result,  # noqa: E501
                                        )
                                    )
                                    FEEDBACK_COLLECTOR.store_compiler_result(
                                        compiler=compiler,
                                        q_type=q_type,
                                        metric_drop_ths=metric_drop_ths,
                                        latency=latency,
                                        compression=None,
                                        pipeline_name=self.pipeline_dl_framework,  # noqa: E501
                                    )
                                else:
                                    self.logger.warning(
                                        "The optimized model will be "
                                        "discarded due to poor results "
                                        "obtained with the given metric."
                                    )
                    except Exception as ex:
                        self.logger.warning(
                            f"Optimization failed with "
                            f"{self.pipeline_dl_framework} "
                            f"interface of {compiler}. Got error {ex}. "
                            f"If possible the compilation will be re-scheduled"
                            f" with another interface. Please consult the "
                            f"documentation for further info or open an issue "
                            f"on GitHub for receiving assistance."
                        )
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            metric_drop_ths=metric_drop_ths,
                            latency=None,
                            compression=None,
                            pipeline_name=self.pipeline_dl_framework,
                        )

    def get_result(self) -> List:
        return self.optimized_models
