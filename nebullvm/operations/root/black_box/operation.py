import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Union,
    Iterable,
    Sequence,
    Dict,
    Callable,
    List,
)

from nebullvm.config import TRAIN_TEST_SPLIT_RATIO
from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.converters import (
    PytorchConverter,
    TensorflowConverter,
    ONNXConverter,
)
from nebullvm.operations.fetch_operations.local import (
    FetchModelFromLocal,
    FetchDataFromLocal,
)
from nebullvm.operations.measures.measures import LatencyOriginalModelMeasure
from nebullvm.operations.measures.utils import QUANTIZATION_METRIC_MAP
from nebullvm.operations.optimizations.optimizers import (
    PytorchOptimizer,
    ONNXOptimizer,
    TensorflowOptimizer,
)
from nebullvm.operations.root.black_box.utils import (
    map_compilers_and_compressors,
    check_input_data,
    get_dl_framework,
    extract_info_from_data,
    is_huggingface_data,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import Module
from nebullvm.tools.base import (
    ModelCompiler,
    DeepLearningFramework,
    ModelParams,
)
from nebullvm.tools.data import DataManager


class BlackBoxModelOptimizationRootOp(Operation):
    def __init__(self):
        super().__init__()

        self.model = None
        self.data = None
        self.optimal_model = None
        self.conversion_op = None

        self.fetch_model_op = FetchModelFromLocal()
        self.fetch_data_op = FetchDataFromLocal()
        self.orig_latency_measure_op = LatencyOriginalModelMeasure()
        self.torch_conversion_op = PytorchConverter()
        self.tensorflow_conversion_op = TensorflowConverter()
        self.torch_optimization_op = PytorchOptimizer()
        self.onnx_conversion_op = ONNXConverter()
        self.onnx_optimization_op = ONNXOptimizer()
        self.tensorflow_optimization_op = TensorflowOptimizer()

    def _get_conversion_op(self, dl_framework: DeepLearningFramework):
        if dl_framework == DeepLearningFramework.PYTORCH:
            conversion_op = self.torch_conversion_op
        elif dl_framework == DeepLearningFramework.TENSORFLOW:
            conversion_op = self.tensorflow_conversion_op
        else:
            conversion_op = self.onnx_conversion_op

        return conversion_op

    def execute(
        self,
        model: Any,
        input_data: Union[Iterable, Sequence],
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
        ignore_compressors: List[str] = None,
        store_latencies: bool = False,
        **kwargs,
    ):
        self.logger.info(
            f"Running Black Box Nebullvm Optimization on {self.device.name}"
        )

        if self.fetch_model_op.get_model() is None:
            self.fetch_model_op.execute(model)
        if self.fetch_data_op.get_data() is None:
            self.fetch_data_op.execute(input_data)

        ignore_compilers = map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        # ignore_compressors = _map_compilers_and_compressors(
        #     ignore_compressors, ModelCompressor
        # )

        # Check availability of model and data
        if self.fetch_model_op.get_model() and self.fetch_data_op.get_data():
            self.model = self.fetch_model_op.get_model()
            self.data = self.fetch_data_op.get_data()

            needs_conversion_to_hf = False
            if is_huggingface_data(self.data[0]):
                (
                    self.model,
                    self.data,
                    input_names,
                    output_structure,
                    output_type,
                ) = convert_hf_model(
                    self.model, self.data, self.device, **kwargs
                )
                needs_conversion_to_hf = True

            if not isinstance(self.data, DataManager):
                if check_input_data(self.data):
                    self.data = DataManager(self.data)
                else:
                    self.data = DataManager.from_iterable(self.data)

            dl_framework = get_dl_framework(self.model)

            if metric_drop_ths is not None and metric_drop_ths <= 0:
                metric_drop_ths = None
            elif metric_drop_ths is not None and metric is None:
                metric = "numeric_precision"
            if isinstance(metric, str):
                metric = QUANTIZATION_METRIC_MAP.get(metric)

            model_params = extract_info_from_data(
                model=self.model,
                input_data=self.data,
                dl_framework=dl_framework,
                dynamic_info=dynamic_info,
                device=self.device,
            )

            self.data.split(TRAIN_TEST_SPLIT_RATIO)

            # Benchmark original model
            self.orig_latency_measure_op.to(self.device).execute(
                model=self.model,
                input_data=self.data.get_split("test"),
                dl_framework=dl_framework,
            )

            with TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir) / "fp32"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                # Convert model to all available frameworks
                self.conversion_op = self._get_conversion_op(dl_framework)
                self.conversion_op.to(self.device).set_state(
                    self.model, self.data
                ).execute(
                    save_path=tmp_dir,
                    model_params=model_params,
                )

                optimized_models = []
                if self.conversion_op.get_result() is not None:
                    original_model_size = (
                        os.path.getsize(self.conversion_op.get_result()[0])
                        if isinstance(self.conversion_op.get_result()[0], str)
                        else len(
                            pickle.dumps(
                                self.conversion_op.get_result()[0], -1
                            )
                        )
                    )
                    for model in self.conversion_op.get_result():
                        optimized_models += self.optimize(
                            model=model,
                            optimization_time=optimization_time,
                            metric_drop_ths=metric_drop_ths,
                            metric=metric,
                            model_params=model_params,
                            ignore_compilers=ignore_compilers,
                            source_dl_framework=dl_framework,
                        )

            optimized_models.sort(key=lambda x: x[1], reverse=False)
            opt_metric_drop = (
                f"{optimized_models[0][2]:.4f}"
                if optimized_models[0][2] > 1e-4
                else "0"
            )

            metric_name = (
                "compute_relative_difference"
                if metric is None
                else metric.__name__
            )

            orig_latency = self.orig_latency_measure_op.get_result()[1]

            self.logger.info(
                (
                    f"\n[ Nebullvm results ]\n"
                    f"Optimization device: {self.device.name}\n"
                    f"Original model latency: {orig_latency:.4f} sec/batch\n"
                    f"Original model throughput: "
                    f"{(1 / orig_latency) * model_params.batch_size:.2f} "
                    f"data/sec\n"
                    f"Original model size: "
                    f"{original_model_size / 1e6:.2f} MB\n"
                    f"Optimized model latency: {optimized_models[0][1]:.4f} "
                    f"sec/batch\n"
                    f"Optimized model throughput: "
                    f"{1 / optimized_models[0][1]:.2f} "
                    f"data/sec\n"
                    f"Optimized model size: "
                    f"{optimized_models[0][0].get_size() / 1e6:.2f} MB\n"
                    f"Optimized model metric drop: {opt_metric_drop} "
                    f"({metric_name})\n"
                    f"Estimated speedup: "
                    f"{orig_latency / optimized_models[0][1]:.2f}x"
                )
            )

            self.optimal_model = optimized_models[0][0]

            if needs_conversion_to_hf:
                from nebullvm.operations.inference_learners.huggingface import (  # noqa: E501
                    HuggingFaceInferenceLearner,
                )

                self.optimal_model = HuggingFaceInferenceLearner(
                    core_inference_learner=self.optimal_model,
                    output_structure=output_structure,
                    input_names=input_names,
                    output_type=output_type,
                )

    def optimize(
        self,
        model: Any,
        optimization_time: str,
        metric_drop_ths: float,
        metric: Callable,
        model_params: ModelParams,
        ignore_compilers: List[str],
        source_dl_framework: DeepLearningFramework,
    ) -> List[BaseInferenceLearner]:
        if self.orig_latency_measure_op.get_result() is not None:
            model_outputs = self.orig_latency_measure_op.get_result()[0]
            if isinstance(model, Module):
                optimization_op = self.torch_optimization_op
            elif isinstance(model, tf.Module) and model is not None:
                optimization_op = self.tensorflow_optimization_op
            else:
                optimization_op = self.onnx_optimization_op

            optimization_op.to(self.device).execute(
                model=model,
                input_data=self.data,
                optimization_time=optimization_time,
                metric_drop_ths=metric_drop_ths,
                metric=metric,
                model_params=model_params,
                model_outputs=model_outputs,
                ignore_compilers=ignore_compilers,
                source_dl_framework=source_dl_framework,
            )

            optimized_models = optimization_op.get_result()

            return optimized_models

    def get_result(self) -> Any:
        return self.optimal_model
