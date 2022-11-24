import abc
from tempfile import TemporaryDirectory
from typing import List

from nebullvm.base import ModelCompiler, QuantizationType
from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.measure import (
    compute_relative_difference,
    compute_optimized_running_time,
)
from nebullvm.operations.base import Operation
from nebullvm.transformations.base import MultiStageTransformation


class Optimizer(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimized_models = []
        self.source_dl_framework = None
        self.pipeline_dl_framework = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _load_compilers(
        self,
        ignore_compilers: List[ModelCompiler],
        metric_drop_ths: float = None,
    ):  # noqa: E501
        raise NotImplementedError()

    def optimize(
        self,
        model,
        input_data,
        optimization_time,
        metric_drop_ths,
        metric,
        model_params,
        model_outputs,
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

        for compiler_op, build_inference_learner_op in zip(
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
                                    self.optimized_models.append(
                                        (
                                            inference_learner,
                                            latency,
                                            self.validity_check_op.measure_result,  # noqa: E501
                                        )
                                    )
                    except Exception as e:
                        # TODO: print error message
                        # raise (e)
                        raise e

    def get_result(self) -> List:
        return self.optimized_models
