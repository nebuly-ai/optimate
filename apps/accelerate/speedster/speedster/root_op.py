import json
import os
import pickle
import sys
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

from loguru import logger
from nebullvm import setup_logger
from nebullvm.config import TRAIN_TEST_SPLIT_RATIO, MIN_NUMBER
from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.converters import (
    PytorchConverter,
    TensorflowConverter,
    ONNXConverter,
)
from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.fetch_operations.local import (
    FetchModelFromLocal,
    FetchDataFromLocal,
)
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.operations.measures.measures import LatencyOriginalModelMeasure
from nebullvm.operations.measures.utils import QUANTIZATION_METRIC_MAP
from nebullvm.operations.optimizations.optimizers import (
    PytorchOptimizer,
    ONNXOptimizer,
    TensorflowOptimizer,
)
from nebullvm.operations.optimizations.utils import (
    map_compilers_and_compressors,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch, DataLoader
from nebullvm.optional_modules.utils import check_dependencies
from nebullvm.tools.base import (
    ModelCompiler,
    DeepLearningFramework,
    ModelParams,
    OptimizationTime,
    ModelCompressor,
    DeviceType,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import (
    get_dl_framework,
    is_huggingface_data,
    check_input_data,
    is_data_subscriptable,
    extract_info_from_data,
)
from tabulate import tabulate

from speedster.utils import (
    get_model_name,
    read_model_size,
    generate_model_id,
    get_hw_info,
)


SPEEDSTER_FEEDBACK_COLLECTOR = FeedbackCollector(
    url="https://nebuly.cloud/v1/store_speedster_results",
    disable_telemetry_environ_var="SPEEDSTER_DISABLE_TELEMETRY",
    app_version="0.2.1",
)


def _convert_technique(technique: str):
    if technique == "none":  # use fp32 instead of none
        technique = "fp32"
    elif technique == "HALF":
        technique = "fp16"
    elif technique == "STATIC":
        technique = "int8"
    else:
        technique = "int8_dynamic"
    return technique


def _get_model_len(model: Any):
    try:
        return len(pickle.dumps(model, -1))
    except Exception:
        logger.warning(
            "Cannot pickle input model. Unable to "
            "extract original model size"
        )
        # Model is not pickable
        return -1


class SpeedsterRootOp(Operation):
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
        self.set_feedback_collector(SPEEDSTER_FEEDBACK_COLLECTOR)

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
        input_data: Union[Iterable, Sequence, DataManager],
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
        device_id = (
            f":{self.device.idx}" if self.device.type is DeviceType.GPU else ""
        )
        self.logger.info(
            f"Running Speedster on {self.device.type.name}{device_id}"
        )

        check_dependencies(self.device)

        if self.fetch_model_op.get_model() is None:
            self.fetch_model_op.execute(model)
        if self.fetch_data_op.get_data() is None:
            self.fetch_data_op.execute(input_data)

        ignore_compilers = map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        ignore_compressors = map_compilers_and_compressors(
            ignore_compressors, ModelCompressor
        )

        optimization_time = OptimizationTime(optimization_time)

        # Check availability of model and data
        if self.fetch_model_op.get_model() and self.fetch_data_op.get_data():
            self.model = self.fetch_model_op.get_model()
            self.data = self.fetch_data_op.get_data()

            if isinstance(self.data, (DataLoader, tf.data.Dataset)):
                try:
                    self.data = DataManager.from_dataloader(self.data)
                except Exception:
                    raise ValueError(
                        "The provided dataloader does not match the expected "
                        "format.\n"
                        "Speedster supports dataloaders that return tuples in "
                        "the\n"
                        "following formats: \n"
                        "Single input: (input,  label)\n"
                        "Multiple inputs: ((input1, input2, ...),  label) or "
                        "(input1, input2, ...,  label)\n"
                        "Inputs and labels should be either tensors or numpy "
                        "arrays,\n"
                        "depending on the framework used.\n"
                    )

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

                if dynamic_info is None:
                    self.logger.warning(
                        "Dynamic shape info has not been provided for the "
                        "HuggingFace model. The resulting optimized model "
                        "will be usable only with a fixed input shape. "
                        "To optimize the model for dynamic shapes, please "
                        "look here: https://nebuly.gitbook.io/nebuly/modules/"
                        "speedster/how-to-guides"
                        "#using-dynamic-shape."
                    )

            if not isinstance(self.data, DataManager):
                if check_input_data(self.data):
                    if is_data_subscriptable(self.data):
                        self.data = DataManager(self.data)
                    else:
                        self.data = DataManager.from_iterable(self.data)
                else:
                    raise ValueError(
                        "The provided data does not match the expected "
                        "format.\n"
                        "Speedster supports data in the following formats: \n"
                        "- PyTorch DataLoader\n"
                        "- TensorFlow Dataset\n"
                        "- List of tuples: [((input_0, ... ), label), ...] \n"
                        "Inputs and labels should be either tensors or numpy "
                        "arrays,\n"
                        "depending on the framework used.\n"
                    )

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

            model_name = get_model_name(self.model)
            model_info = {
                "model_name": model_name,
                "model_size": read_model_size(model),
                "framework": dl_framework.value,
            }
            self.feedback_collector.store_info(
                key="model_id", value=generate_model_id(model_name)
            )
            self.feedback_collector.store_info(
                key="model_metadata", value=model_info
            )
            self.feedback_collector.store_info(
                key="hardware_setup", value=get_hw_info(self.device)
            )

            # Benchmark original model
            self.orig_latency_measure_op.to(self.device).execute(
                model=self.model,
                input_data=self.data.get_split("test"),
                dl_framework=dl_framework,
            )

            # Store original model result
            original_model_dict = {
                "compiler": dl_framework.value,
                "technique": "original",
                "latency": self.orig_latency_measure_op.get_result()[1],
            }

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
                        else _get_model_len(self.conversion_op.get_result()[0])
                    )
                    for model in self.conversion_op.get_result():
                        optimized_models += self._optimize(
                            model=model,
                            optimization_time=optimization_time,
                            metric_drop_ths=metric_drop_ths,
                            metric=metric,
                            model_params=model_params,
                            ignore_compilers=ignore_compilers,
                            ignore_compressors=ignore_compressors,
                            source_dl_framework=dl_framework,
                        )

            optimized_models.sort(key=lambda x: x[1], reverse=False)

            if len(optimized_models) < 1 or optimized_models[0][0] is None:
                self.logger.warning(
                    "No optimized model has been created. This is likely "
                    "due to a bug in Speedster. Please open an issue and "
                    "report in details your use case."
                )
            else:
                opt_metric_drop = (
                    f"{optimized_models[0][2]:.4f}"
                    if optimized_models[0][2] > MIN_NUMBER
                    else "0"
                )

                orig_latency = self.orig_latency_measure_op.get_result()[1]

                optimizations = self.feedback_collector.get("optimizations")
                valid_optimizations = [
                    v for v in optimizations if v["latency"] != -1
                ]
                best_technique = _convert_technique(
                    sorted(valid_optimizations, key=lambda x: x["latency"])[0][
                        "technique"
                    ]
                )
                optimizations.insert(0, original_model_dict)
                self.feedback_collector.send_feedback()
                if store_latencies:
                    model_id = self.feedback_collector.get("model_id", "")
                    with open(
                        f"{model_name}_latencies_{model_id[:10]}.json", "w"
                    ) as f:
                        json.dump(
                            {
                                "optimizations": optimizations,
                            },
                            f,
                        )
                self.feedback_collector.reset("optimizations")
                self.feedback_collector.reset("model_id")
                self.feedback_collector.reset("model_metadata")

                table = [
                    [
                        "backend",
                        dl_framework.name,
                        optimized_models[0][0].name,
                        "",
                    ],
                    [
                        "latency",
                        f"{orig_latency:.4f} sec/batch",
                        f"{optimized_models[0][1]:.4f} sec/batch",
                        f"{orig_latency / optimized_models[0][1]:.2f}x",
                    ],
                    [
                        "throughput",
                        f"{(1 / orig_latency) * model_params.batch_size:.2f} "
                        f"data/sec",
                        f"{1 / optimized_models[0][1]:.2f} data/sec",
                        f"{(1 / optimized_models[0][1]) / (1 / orig_latency):.2f}x",  # noqa: E501
                    ],
                    [
                        "model size",
                        f"{original_model_size / 1e6:.2f} MB",
                        f"{optimized_models[0][0].get_size() / 1e6:.2f} MB",
                        f"{min(int((optimized_models[0][0].get_size()-original_model_size) / original_model_size * 100), 0)}%"  # noqa: E501
                        if original_model_size > 0
                        else "NA",
                    ],
                    ["metric drop", "", opt_metric_drop, ""],
                    [
                        "techniques",
                        "",
                        f"{best_technique}",
                        "",
                    ],
                ]
                headers = [
                    "Metric",
                    "Original Model",
                    "Optimized Model",
                    "Improvement",
                ]

                # change format to the logger, avoiding printing verbose info
                # to the console (as date, time, etc.)
                self.logger.remove()
                handler_id = self.logger.add(
                    sys.stdout, format="<level>{message}</level>"
                )
                hw_info = get_hw_info(self.device)
                hw_name = hw_info[
                    "cpu" if self.device.type is DeviceType.CPU else "gpu"
                ]
                self.logger.info(
                    (
                        f"\n[Speedster results on {hw_name}]\n"
                        f"{tabulate(table, headers, tablefmt='heavy_outline')}"
                    )
                )

                if orig_latency / optimized_models[0][1] < 2:
                    # if self.device.type is DeviceType.CPU:
                    #     device = hw_info["cpu"]
                    # else:
                    #     device = hw_info["gpu"]

                    self.logger.warning(
                        f"\nMax speed-up with your input parameters is "
                        f"{orig_latency / optimized_models[0][1]:.2f}x. "
                        f"If you want to get a faster optimized model, "
                        f"see the following link for some suggestions: "
                        f"https://docs.nebuly.com/modules/speedster/getting-"
                        f"started/run-the-optimization#acceleration-sugges"
                        f"tions\n"
                    )

                self.logger.remove(handler_id)
                setup_logger()

                if needs_conversion_to_hf:
                    from nebullvm.operations.inference_learners.huggingface import (  # noqa: E501
                        HuggingFaceInferenceLearner,
                    )

                    self.optimal_model = HuggingFaceInferenceLearner(
                        core_inference_learner=optimized_models[0][0],
                        output_structure=output_structure,
                        input_names=input_names,
                        output_type=output_type,
                    )
                else:
                    self.optimal_model = optimized_models[0][0]

    def _optimize(
        self,
        model: Any,
        optimization_time: OptimizationTime,
        metric_drop_ths: float,
        metric: Callable,
        model_params: ModelParams,
        ignore_compilers: List[ModelCompiler],
        ignore_compressors: List[ModelCompressor],
        source_dl_framework: DeepLearningFramework,
    ) -> List[BaseInferenceLearner]:
        if self.orig_latency_measure_op.get_result() is not None:
            model_outputs = self.orig_latency_measure_op.get_result()[0]
            if isinstance(model, torch.nn.Module):
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
                ignore_compressors=ignore_compressors,
                source_dl_framework=source_dl_framework,
            )

            optimized_models = optimization_op.get_result()

            if isinstance(model, torch.nn.Module):
                optimization_op.free_model_gpu(model)

            return optimized_models

    def get_result(self) -> Any:
        return self.optimal_model
