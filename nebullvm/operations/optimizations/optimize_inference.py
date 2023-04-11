from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Callable, List, Union, Dict, Optional

from nebullvm.config import TRAIN_TEST_SPLIT_RATIO
from nebullvm.core import types
from nebullvm.core.models import (
    OptimizeInferenceResult,
    OriginalModel,
    OptimizedModel,
    BenchmarkOriginalModelResult,
    ModelCompiler,
    ModelCompressor,
    OptimizationTime,
    ModelParams,
    DeepLearningFramework,
)
from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.utils import get_conversion_op
from nebullvm.operations.measures.measures import LatencyOriginalModelMeasure
from nebullvm.operations.measures.utils import QUANTIZATION_METRIC_MAP
from nebullvm.operations.optimizations.optimizers.optimizers import (
    PytorchOptimizer,
    TensorflowOptimizer,
    ONNXOptimizer,
)
from nebullvm.operations.optimizations.utils import (
    map_compilers_and_compressors,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import DataLoader as TorchDataLoader
from nebullvm.optional_modules.torch import torch
from nebullvm.optional_modules.utils import (
    check_dependencies,
)
from nebullvm.tools.adapters import (
    ModelAdapter,
    DiffusionAdapter,
    HuggingFaceAdapter,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.diffusers import (
    is_diffusion_model_pipe,
    is_diffusion_model,
)
from nebullvm.tools.hardware_utils import get_hw_setup
from nebullvm.tools.utils import (
    is_huggingface_data,
    check_input_data,
    is_data_subscriptable,
    get_dl_framework,
    extract_info_from_data,
    get_model_name,
    get_model_size_mb,
    get_throughput,
)


class OptimizeInferenceOp(Operation):
    def __init__(self):
        super().__init__()
        self.torch_optimization_op = PytorchOptimizer()
        self.onnx_optimization_op = ONNXOptimizer()
        self.tensorflow_optimization_op = TensorflowOptimizer()

    @staticmethod
    def _as_data_manager(data) -> DataManager:
        if isinstance(data, DataManager):
            return data
        if check_input_data(data) is False:
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
        if is_data_subscriptable(data):
            return DataManager(data)
        else:
            return DataManager.from_iterable(data)

    @staticmethod
    def _check_inputs(model: Any, input_data: types.InputData):
        if model is None:
            raise ValueError("Input model cannot be None")
        if len(input_data) == 0:
            raise ValueError("Input data cannot be empty")

    def execute(
        self,
        model: Any,
        input_data: types.InputData,
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
        ignore_compressors: List[str] = None,
        store_latencies: bool = False,
        **kwargs,
    ) -> OptimizeInferenceResult:

        self._check_inputs(model, input_data)
        check_dependencies(self.device)

        ignore_compilers = map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        ignore_compressors = map_compilers_and_compressors(
            ignore_compressors, ModelCompressor
        )

        optimization_time = OptimizationTime(optimization_time)

        data = input_data

        if isinstance(data, (TorchDataLoader, tf.data.Dataset)):
            try:
                data = DataManager.from_dataloader(data)
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

        # Setup adapters
        model_adapter: Optional[ModelAdapter] = None
        if is_diffusion_model_pipe(model):
            self.logger.info(
                "The provided model is a diffusion model. "
                "Speedster will optimize the UNet part of the model."
            )
            model_adapter = DiffusionAdapter(model, data, self.device)
        elif is_huggingface_data(data[0]):
            model_adapter = HuggingFaceAdapter(
                model, data, self.device, **kwargs
            )
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

        # Adapt data and model
        if model_adapter is not None:
            data = model_adapter.adapted_data
            model = model_adapter.adapted_model

        data = self._as_data_manager(data)
        dl_framework = get_dl_framework(model)

        if metric_drop_ths is not None and metric_drop_ths <= 0:
            metric_drop_ths = None
        elif metric_drop_ths is not None and metric is None:
            metric = "numeric_precision"
        if isinstance(metric, str):
            metric = QUANTIZATION_METRIC_MAP.get(metric)

        model_params: ModelParams = extract_info_from_data(
            model=model,
            input_data=data,
            dl_framework=dl_framework,
            dynamic_info=dynamic_info,
            device=self.device,
            is_diffusion=is_diffusion_model(model),
        )

        data.split(TRAIN_TEST_SPLIT_RATIO)

        # -------- Benchmark original model --------
        original_latency_op = LatencyOriginalModelMeasure().to(self.device)
        orig_model_benchmark: BenchmarkOriginalModelResult = (
            original_latency_op.execute(
                model=model,
                input_data=data.get_split("test"),
                dl_framework=dl_framework,
            )
        )
        original_model = OriginalModel(
            model=model,
            latency_seconds=orig_model_benchmark.latency_seconds,
            name=get_model_name(model),
            size_mb=get_model_size_mb(model),
            framework=dl_framework,
            throughput=get_throughput(
                latency=orig_model_benchmark.latency_seconds,
                # Normal models have batch size B, diffusion
                # models have batch size 2B
                batch_size=model_params.batch_size
                if not is_diffusion_model(model)
                else model_params.batch_size / 2,
            ),
        )
        # ------------------------------------------

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir) / "fp32"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Convert model to all available frameworks
            conversion_op = get_conversion_op(dl_framework)
            conversion_op.to(self.device).set_state(model, data).execute(
                save_path=tmp_dir,
                model_params=model_params,
            )

            # Optimize models
            optimized_models: List[OptimizedModel] = []
            is_diffusion = is_diffusion_model(model)
            for i, model in enumerate(conversion_op.get_result()):
                optimized_models += self._optimize(
                    model=model,
                    input_data=data,
                    model_outputs=orig_model_benchmark.model_outputs,
                    optimization_time=optimization_time,
                    metric_drop_ths=metric_drop_ths,
                    metric=metric,
                    model_params=model_params,
                    ignore_compilers=ignore_compilers,
                    ignore_compressors=ignore_compressors,
                    source_dl_framework=dl_framework,
                    pipeline_idx=i + 1,
                    len_pipelines=len(conversion_op.get_result()),
                    is_diffusion=is_diffusion,
                )

        optimized_models.sort(key=lambda x: x.latency_seconds, reverse=False)

        # Check if at least one optimized model has been created
        no_optimized_models = len(optimized_models) < 1
        no_inference_learners = all(
            o.inference_learner is None for o in optimized_models
        )
        if no_optimized_models or no_inference_learners:
            self.logger.warning(
                "No optimized model has been created. This is likely "
                "due to a bug during optimization. Please open an issue "
                "and report in details your use case."
            )

        # Extract lowest-latency model
        lowest_latency = self._extract_lowest_latency_model(optimized_models)

        if model_adapter is not None:
            original_model = model_adapter.adapt_original_model(original_model)
            lowest_latency = model_adapter.adapt_inference_learner(
                lowest_latency
            )

        return OptimizeInferenceResult(
            original_model=original_model,
            optimized_model=lowest_latency,
            hardware_setup=get_hw_setup(),
        )

    def _optimize(
        self,
        model: Any,
        model_outputs: Iterable,
        input_data: types.InputData,
        optimization_time: OptimizationTime,
        metric_drop_ths: float,
        metric: Callable,
        model_params: ModelParams,
        ignore_compilers: List[ModelCompiler],
        ignore_compressors: List[ModelCompressor],
        source_dl_framework: DeepLearningFramework,
        pipeline_idx: int,
        len_pipelines: int,
        is_diffusion: bool,
    ) -> List[OptimizedModel]:
        if isinstance(model, torch.nn.Module):
            optimization_op = self.torch_optimization_op
            self.logger.info(
                f"[{pipeline_idx}/{len_pipelines}] Running PyTorch "
                f"Optimization Pipeline"
            )
        elif isinstance(model, tf.Module):
            optimization_op = self.tensorflow_optimization_op
            self.logger.info(
                f"[{pipeline_idx}/{len_pipelines}] Running TensorFlow "
                f"Optimization Pipeline"
            )
        else:
            optimization_op = self.onnx_optimization_op
            self.logger.info(
                f"[{pipeline_idx}/{len_pipelines}] Running ONNX "
                f"Optimization Pipeline"
            )

        # Run optimization
        optimized_models = optimization_op.to(self.device).execute(
            model=model,
            input_data=input_data,
            optimization_time=optimization_time,
            metric_drop_ths=metric_drop_ths,
            metric=metric,
            model_params=model_params,
            model_outputs=model_outputs,
            ignore_compilers=ignore_compilers,
            ignore_compressors=ignore_compressors,
            source_dl_framework=source_dl_framework,
            is_diffusion=is_diffusion,
        )

        if isinstance(model, torch.nn.Module):
            optimization_op.free_model_gpu(model)

        return optimized_models

    @staticmethod
    def _extract_lowest_latency_model(
        models: List[OptimizedModel],
    ) -> Optional[OptimizedModel]:
        # fmt: off
        inference_learner_models = [
            m for m in models
            if m.inference_learner is not None
        ]
        # fmt: on
        if len(inference_learner_models) == 0:
            return None
        return min(inference_learner_models, key=lambda m: m.latency_seconds)
