from typing import List

from nebullvm.base import QuantizationType
from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.measure import (
    compute_relative_difference,
    compute_optimized_running_time,
)
from nebullvm.operations.inference_learners.builders import (
    PytorchBuildInferenceLearner,
)
from nebullvm.operations.measures.measures import PrecisionMeasure
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.pytorch import (
    PytorchBackendCompiler,
)
from nebullvm.transformations.base import MultiStageTransformation


class PytorchOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.compiler_op = PytorchBackendCompiler()
        self.build_inference_learner_op = PytorchBuildInferenceLearner()
        self.validity_check_op = PrecisionMeasure()

    def _get_compilers(self) -> List[Compiler]:
        pass

    def execute(
        self,
        model,
        input_data,
        optimization_time,
        metric_drop_ths,
        metric,
        model_params,
        model_outputs,
        device,
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

        input_tfms = MultiStageTransformation([])
        self.optimized_models = []

        # TODO: extend this to call all compressors and all compilers
        for q_type in q_types:
            try:
                self.compiler_op.execute(
                    model=model,
                    input_data=input_data,
                    device=device,
                    metric_drop_ths=metric_drop_ths
                    if q_type is not None
                    else None,
                    quantization_type=q_type,
                    input_tfms=input_tfms,
                )

                compiled_model = self.compiler_op.compiled_model
                self.build_inference_learner_op.execute(
                    compiled_model, model_params, device
                )
                inference_learner = (
                    self.build_inference_learner_op.inference_learner
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
                            f"Optimized model latency: {latency} sec/iter"
                        )
                        self.optimized_models.append(
                            (
                                inference_learner,
                                latency,
                                self.validity_check_op.measure_result,
                            )
                        )
            except Exception:
                # TODO: print error message
                continue


class TensorflowOptimizer(Optimizer):
    def _get_compilers(self) -> List[Compiler]:
        pass

    def execute(self, **kwargs):
        pass


class OnnxOptimizer(Optimizer):
    def _get_compilers(self) -> List[Compiler]:
        pass

    def execute(self, **kwargs):
        pass
