import json
import pickle
import sys
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
from nebullvm.config import MIN_NUMBER
from nebullvm.core.models import OptimizeInferenceResult, DeviceType
from nebullvm.operations.base import Operation
from nebullvm.operations.optimizations.optimize_inference import (
    OptimizeInferenceOp,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.feedback_collector import FeedbackCollector
from tabulate import tabulate

from nebullvm.tools.hardware_utils import get_hw_setup
from nebullvm.tools.utils import (
    get_model_size_mb,
    get_model_name,
    generate_model_id,
)

SPEEDSTER_FEEDBACK_COLLECTOR = FeedbackCollector(
    url="https://nebuly.cloud/v1/store_speedster_results",
    disable_telemetry_environ_var="SPEEDSTER_DISABLE_TELEMETRY",
    app_version="0.4.0",
)


def _convert_technique(technique: str):
    if technique.lower() == "none":  # use fp32 instead of none
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
        self.optimize_inference_op = OptimizeInferenceOp()
        self.set_feedback_collector(SPEEDSTER_FEEDBACK_COLLECTOR)

    def _send_feedback(
        self,
        optimization_result: OptimizeInferenceResult,
        store_latencies: bool = False,
    ):
        model_orig = optimization_result.original_model.model
        model_name = get_model_name(model_orig)
        model_info = {
            "model_name": model_name,
            "model_size": f"{get_model_size_mb(model_orig)} MB",
            "framework": optimization_result.original_model.framework.value,
        }
        self.feedback_collector.store_info(
            key="model_id", value=generate_model_id(model_orig)
        )
        self.feedback_collector.store_info(
            key="model_metadata", value=model_info
        )
        self.feedback_collector.store_info(
            key="hardware_setup", value=get_hw_setup(self.device).__dict__
        )
        optimizations = self.feedback_collector.get("optimizations")
        original_model_dict = {
            "compiler": optimization_result.original_model.framework.value,
            "technique": "original",
            "latency": optimization_result.original_model.latency_seconds,
        }
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
        self.logger.info(
            "Running Speedster on {}{}".format(
                self.device.type.name,
                f":{self.device.idx}"
                if self.device.type is not DeviceType.CPU
                else "",
            )
        )

        result = self.optimize_inference_op.to(self.device).execute(
            model=model,
            input_data=input_data,
            metric_drop_ths=metric_drop_ths,
            metric=metric,
            optimization_time=optimization_time,
            dynamic_info=dynamic_info,
            config_file=config_file,
            ignore_compilers=ignore_compilers,
            ignore_compressors=ignore_compressors,
            store_latencies=store_latencies,
            **kwargs,
        )

        if result.optimized_model is None:
            return None

        opt_metric_drop = (
            f"{result.metric_drop:.4f}"
            if result.metric_drop > MIN_NUMBER
            else "0"
        )

        self._send_feedback(result, store_latencies=store_latencies)

        table = [
            [
                "backend",
                result.original_model.framework.name,
                result.optimized_model.inference_learner.name,
                "",
            ],
            [
                "latency",
                f"{result.original_model.latency_seconds:.4f} sec/batch",
                f"{result.optimized_model.latency_seconds:.4f} sec/batch",
                f"{result.original_model.latency_seconds / result.optimized_model.latency_seconds:.2f}x",  # noqa: E501
            ],
            [
                "throughput",
                f"{result.original_model.throughput:.2f} " f"data/sec",
                f"{result.optimized_model.throughput:.2f} " f"data/sec",
                f"{result.optimized_model.throughput / result.original_model.throughput:.2f}x",  # noqa: E501
            ],
            [
                "model size",
                f"{result.original_model.size_mb:.2f} MB",
                f"{result.optimized_model.size_mb:.2f} MB",
                f"{min(int((result.optimized_model.size_mb-result.original_model.size_mb) / result.original_model.size_mb * 100), 0)}%"  # noqa: E501
                if result.original_model.size_mb > 0
                else "NA",
            ],
            ["metric drop", "", opt_metric_drop, ""],
            [
                "techniques",
                "",
                f"{_convert_technique(result.optimized_model.technique)}",
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
        hw_info = get_hw_setup(self.device)
        hw_name = (
            hw_info.cpu
            if self.device.type is DeviceType.CPU
            else hw_info.accelerator
        )
        self.logger.info(
            (
                f"\n[Speedster results on {hw_name}]\n"
                f"{tabulate(table, headers, tablefmt='heavy_outline')}"
            )
        )

        if (
            result.original_model.latency_seconds
            / result.optimized_model.latency_seconds
            < 2
        ):
            self.logger.warning(
                f"\nMax speed-up with your input parameters is "
                f"{result.original_model.latency_seconds / result.optimized_model.latency_seconds:.2f}x. "  # noqa: E501
                f"If you want to get a faster optimized model, "
                f"see the following link for some suggestions: "
                f"https://docs.nebuly.com/Speedster/advanced_"
                f"options/#acceleration-suggestions\n"
            )

        self.logger.remove(handler_id)
        setup_logger()

        return result.optimized_model.inference_learner
