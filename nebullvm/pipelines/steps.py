import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Tuple, Optional

import cpuinfo
import numpy as np
import torch.nn
from tqdm import tqdm

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    QuantizationType,
    ModelCompiler,
    ModelCompressor,
    OptimizationTime,
)
from nebullvm.compressors.base import BaseCompressor
from nebullvm.compressors.intel import TorchIntelPruningCompressor
from nebullvm.compressors.sparseml import SparseMLCompressor
from nebullvm.config import MIN_DIM_INPUT_DATA
from nebullvm.inference_learners import (
    BaseInferenceLearner,
    PytorchBaseInferenceLearner,
)
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    BaseOptimizer,
    ApacheTVMOptimizer,
    COMPILER_TO_OPTIMIZER_MAP,
    DeepSparseOptimizer,
)
from nebullvm.optimizers.blade_disc import BladeDISCOptimizer
from nebullvm.optimizers.neural_compressor import NeuralCompressorOptimizer
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer
from nebullvm.optimizers.tensorflow import TensorflowBackendOptimizer
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.compilers import (
    tvm_is_available,
    select_compilers_from_hardware_onnx,
    deepsparse_is_available,
    bladedisc_is_available,
    torch_tensorrt_is_available,
    intel_neural_compressor_is_available,
)
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.utils.general import is_python_version_3_10

logger = logging.getLogger("nebullvm_logger")


class Step(ABC):
    """Fundamental building block for the Pipeline."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict:
        """Run the pipeline step."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()


class CompressorStep(Step, ABC):
    """Object managing the Compressor step in the Pipeline. This step manages
    all the defined Compressor objects available considering the data given
    by the user.

    Attributes:
        config_file (str, optional): The configuration file containing the
            configuration parameter for each Compressor. The config_file is
            a YAML file having as main keywords the Compressor names and as
            values dictionaries containing the specific parameters for the
            related Compressor object.
    """

    def __init__(self, config_file: str = None):
        super().__init__()
        self._config_file = config_file

    def run(
        self,
        model: Any = None,
        input_data: DataManager = None,
        metric_drop_ths: float = None,
        metric: Callable = None,
        ignore_compressors: List[ModelCompressor] = None,
        **kwargs,
    ) -> Dict:
        """Run the CompressorStep.

        Args:
            model (Any): Model to be compressed.
            input_data (DataManager): Data to be used for compressing the
                model.
            metric_drop_ths: Maximum reduction in the selected metric accepted.
                No model with a higher error will be accepted. Note that the
                maximum error is modified and then propagated to the next
                steps.
            metric (Callable): Metric to be used for estimating the error
                due to the compression.
            ignore_compressors (List, optional): List of compressors
                to be ignored.
            kwargs (Dict): Keyword arguments propagated to the next step.
        """
        compressor_dict = self._get_compressors(ignore_compressors)
        logger.info(f"Compressions: {tuple(compressor_dict.keys())}")
        models = {"no_compression": (copy.deepcopy(model), metric_drop_ths)}

        # input_data[0][0][0].shape[0] is the batch size
        if (
            len(input_data) < 1
            or (len(input_data) * input_data[0][0][0].shape[0])
            < MIN_DIM_INPUT_DATA
        ):
            logger.warning(
                f"Compression step needs at least {MIN_DIM_INPUT_DATA} "
                f"input data to be activated. Please provide more inputs. "
                f"Compression step will be skipped."
            )
        else:
            train_input_data = input_data.get_split("train")
            eval_input_data = input_data.get_split("eval")
            if len(eval_input_data) > 0:
                for technique, compressor in tqdm(compressor_dict.items()):
                    try:
                        compressed_model, ths = compressor.compress(
                            model,
                            train_input_data,
                            eval_input_data,
                            metric_drop_ths,
                            metric,
                        )
                        models[technique] = (compressed_model, ths)
                    except Exception as ex:
                        logger.warning(
                            f"Error during compression {technique}. Got error "
                            f"{ex}. The compression technique will be skipped."
                            f" Please consult the documentation for further "
                            f"info or open an issue on GitHub for receiving "
                            f"assistance."
                        )
            else:
                logger.warning(
                    "Please note that DIM_DATASET / BATCH_SIZE >= 5, "
                    "otherwise the data cannot be split in training and "
                    "evaluation set properly. Compression step will be "
                    "skipped."
                )
        return {
            "models": models,
            "input_data": input_data,
            "metric": metric,
            "ignore_compressors": ignore_compressors,
            **kwargs,
        }

    @abstractmethod
    def _get_compressors(
        self, ignore_compressors
    ) -> Dict[str, BaseCompressor]:
        raise NotImplementedError()

    @property
    def name(self):
        return "compression_step"


class TorchCompressorStep(CompressorStep):
    """Object managing the Compressor step in the Pipeline for PyTorch models.
    This step manages all the defined Compressor objects available considering
    the data given by the user.

    At the current state this step supports pruning with SparseML and (just on
    intel devices) pruning with the IntelNeuralCompressor.

    Attributes:
        config_file (str, optional): The configuration file containing the
            configuration parameter for each Compressor. The config_file is
            a YAML file having as main keywords the Compressor names and as
            values dictionaries containing the specific parameters for the
            related Compressor object.
    """

    def _get_compressors(
        self, ignore_compressors
    ) -> Dict[str, BaseCompressor]:
        compressors = {}

        if (
            deepsparse_is_available()
            and not is_python_version_3_10()
            and ModelCompressor.SPARSEML not in ignore_compressors
        ):
            compressors["sparseml"] = SparseMLCompressor(
                config_file=self._config_file
            )

        if (
            "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower()
            and ModelCompressor.NEURAL_COMPRESSOR_PRUNING
            not in ignore_compressors
        ):
            compressors["intel_pruning"] = TorchIntelPruningCompressor(
                config_file=self._config_file
            )
        return compressors


class NoCompressionStep(Step):
    """Step to be used when no compression is required."""

    def run(
        self, model: Any, metric_drop_ths: Optional[float], **kwargs
    ) -> Dict:
        return {
            "models": {"no_compression": (model, metric_drop_ths)},
            **kwargs,
        }

    @property
    def name(self):
        return "no_compression"


class OptimizerStep(Step, ABC):
    """Object managing the Optimizers in the pipeline step. All available
    optimizers are run on the model given as input and a list of tuples
    (optimized_model, latency) is given as output.

    """

    def run(
        self,
        models: Dict[str, Tuple[Any, Optional[float]]] = None,
        output_library: DeepLearningFramework = None,
        model_params: ModelParams = None,
        input_tfms: MultiStageTransformation = None,
        metric: Optional[Callable] = None,
        input_data: Optional[DataManager] = None,
        ignore_compilers: List[ModelCompiler] = None,
        optimization_time: OptimizationTime = None,
        pipeline_name: str = None,
        model_outputs: Any = None,
        **kwargs,
    ) -> Dict:
        """Run the OptimizerStep for all the available compilers.

        Args:
            models (Dict): Dictionary of models produced by the CompressorStep.
                For each model produced by the previous step, the updated
                metric_drop_ths (i.e. the error allowed on the model) is
                given together with the model. Keys represent the compression
                technique used for obtaining the model.
            output_library (DeepLearningFramework): The target framework.
            model_params (ModelParams): The model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric (Callable): Metric to be used for estimating the error
                due to the compression.
            input_data (DataManager): Input data to be used for optimizing the
                model.
            ignore_compilers (List): List of compilers to be ignored.
            optimization_time (OptimizationTime): The optimization time mode.
                It can be either 'constrained' or 'unconstrained'. For
                'unconstrained' optimization all the compilers are re-used on
                the different framework interfaces, even if the model has
                already been compiled with the same compiler on another
                framework interface.
            pipeline_name (str): Name of the pipeline.
            model_outputs (Any): Outputs computed by the original model
            kwargs (Dict): Extra keywords that will be ignored.
        """

        optimizers = self._get_optimizers(ignore_compilers)
        logger.info(
            f"Optimizations: "
            f"{tuple(compiler.value for compiler in optimizers.keys())}"
        )
        optimized_models = []

        for prev_tech, (model, metric_drop_ths) in tqdm(models.items()):
            logger.info(f"Optimizing output of {prev_tech}")
            if model is None:
                continue
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
            for compiler, optimizer in tqdm(optimizers.items()):
                for q_type in q_types:
                    try:
                        optimized_model = self._run_optimizer(
                            optimizer,
                            model,
                            output_library,
                            model_params,
                            input_tfms.copy(),
                            metric_drop_ths,
                            q_type,
                            metric,
                            input_data,
                            model_outputs,
                        )
                        if optimized_model is not None:
                            latency = compute_optimized_running_time(
                                optimized_model, input_data
                            )
                        else:
                            latency = np.inf
                        optimized_models.append((optimized_model, latency))
                        if (
                            compiler not in ignore_compilers
                            and optimization_time
                            is OptimizationTime.CONSTRAINED
                        ):
                            ignore_compilers.append(compiler)
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            metric_drop_ths=metric_drop_ths,
                            latency=latency,
                            compression=prev_tech,
                            pipeline_name=pipeline_name,
                        )
                    except Exception as ex:
                        logger.warning(
                            f"Compilation failed with {output_library.value} "
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
                            compression=prev_tech,
                            pipeline_name=pipeline_name,
                        )

        return {
            "optimized_models": optimized_models,
            "ignore_compilers": ignore_compilers,
        }

    @property
    def name(self):
        return "optimizer_step"

    @abstractmethod
    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outpust: Any = None,
    ) -> BaseInferenceLearner:
        raise NotImplementedError()

    @abstractmethod
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        raise NotImplementedError()


class TorchOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting PyTorch
    as compiler interface. All available optimizers are run on the model given
    as input and a list of tuples (optimized_model, latency) is given as
    output.

    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {}
        if ModelCompiler.TORCHSCRIPT not in ignore_compilers:
            optimizers[ModelCompiler.TORCHSCRIPT] = PytorchBackendOptimizer()
        if (
            tvm_is_available()
            and ModelCompiler.APACHE_TVM not in ignore_compilers
        ):
            optimizers[ModelCompiler.APACHE_TVM] = ApacheTVMOptimizer()
        if (
            deepsparse_is_available()
            and ModelCompiler.DEEPSPARSE not in ignore_compilers
        ):
            optimizers[ModelCompiler.DEEPSPARSE] = DeepSparseOptimizer()
        if (
            bladedisc_is_available()
            and ModelCompiler.BLADEDISC not in ignore_compilers
        ):
            optimizers[ModelCompiler.BLADEDISC] = BladeDISCOptimizer()
        if (
            torch_tensorrt_is_available()
            and ModelCompiler.TENSOR_RT not in ignore_compilers
        ):
            optimizers[ModelCompiler.TENSOR_RT] = TensorRTOptimizer()
        if (
            intel_neural_compressor_is_available()
            and ModelCompiler.INTEL_NEURAL_COMPRESSOR not in ignore_compilers
        ):
            optimizers[
                ModelCompiler.INTEL_NEURAL_COMPRESSOR
            ] = NeuralCompressorOptimizer()
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_torch"):
            optimized_model = optimizer.optimize_from_torch(
                torch_model=model,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
                model_outputs=model_outputs,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
                model_outputs=model_outputs,
            )
        return optimized_model


class TFOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting
    TensorFlow as compiler interface. All available optimizers are run on
    the model given as input and a list of tuples (optimized_model, latency)
    is given as output.

    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {}
        if ModelCompiler.TFLITE not in ignore_compilers:
            optimizers[ModelCompiler.TFLITE] = TensorflowBackendOptimizer()
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_tf"):
            optimized_model = optimizer.optimize_from_tf(
                torch_model=model,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
                model_outputs=model_outputs,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
                model_outputs=model_outputs,
            )
        return optimized_model


class OnnxOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting ONNX
    as compiler interface. All available optimizers are run on the model given
    as input and a list of tuples (optimized_model, latency) is given as
    output.

    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        compilers = select_compilers_from_hardware_onnx()
        optimizers = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler]()
            for compiler in compilers
            if compiler not in ignore_compilers
        }
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> PytorchBaseInferenceLearner:
        optimized_model = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            metric_drop_ths=metric_drop_ths
            if quantization_type is not None
            else None,
            metric=metric,
            quantization_type=quantization_type,
            input_tfms=input_tfms,
            input_data=input_data,
            model_outputs=model_outputs,
        )
        return optimized_model


class Pipeline(Step):
    """Pipeline object.

    A Pipeline is a list of steps executed sequentially, where each step
    takes as input the output of the previous one.

    Attributes:
        pipeline_name: str,
        steps (List): List of Steps composing the pipeline.
    """

    def __init__(self, pipeline_name: str, steps: List[Step]):
        super().__init__()
        self._name = pipeline_name
        self._steps = steps

    def run(self, **kwargs) -> Dict:
        logger.info(f"Running pipeline: {self.name}")
        kwargs["pipeline_name"] = self.name.split("_")[0]
        for step in self._steps:
            logger.info(f"Running step: {step.name}")
            kwargs = step.run(**kwargs)
        return kwargs

    @property
    def name(self):
        return self._name


def _get_compressor_step(
    model: Any,
    optimization_time: OptimizationTime,
    config_file: Optional[str],
    metric_drop_ths: Optional[float],
) -> Step:
    if optimization_time is OptimizationTime.CONSTRAINED:
        return NoCompressionStep()
    if metric_drop_ths is None:
        return NoCompressionStep()
    elif isinstance(model, torch.nn.Module):
        return TorchCompressorStep(config_file=config_file)
    else:  # default is NoCompression
        return NoCompressionStep()


def _get_optimizer_step(
    model: Any,
) -> Step:
    if isinstance(model, torch.nn.Module):
        return TorchOptimizerStep()
    elif isinstance(model, tf.Module) and model is not None:
        return TFOptimizerStep()
    else:
        return OnnxOptimizerStep()


def _get_pipeline_name(model: Any):
    if isinstance(model, torch.nn.Module):
        return "pytorch_pipeline"
    elif isinstance(model, tf.Module) and model is not None:
        return "tensorflow_pipeline"
    else:
        return "onnx_pipeline"


def build_pipeline_from_model(
    model: Any,
    optimization_time: OptimizationTime,
    metric_drop_ths: Optional[float],
    config_file: Optional[str],
) -> Pipeline:
    """Function for building a pipeline from a model and user-defined
    parameters

    Args:
        model (Any): The input model.
        optimization_time (OptimizationTime): The optimization time mode.
            It can be either 'constrained' or 'unconstrained'. For
            'constrained' mode just compilers and precision reduction
            techniques are used (no compression). 'Unconstrained' optimization
            allows the usage of more time consuming techniques as pruning and
            distillation.
        metric_drop_ths (float, optional): Maximum reduction in the
            selected metric accepted. No model with an higher error will be
            accepted, i.e. all optimized model having a larger error respect to
            the original one will be discarded, without even considering their
            possible speed-up.
        config_file (str, optional): Configuration file containing the
            parameters needed for defining the CompressionStep in the pipeline.
    """
    compressor_step = _get_compressor_step(
        model, optimization_time, config_file, metric_drop_ths
    )
    optimizer_step = _get_optimizer_step(model)
    pipeline = Pipeline(
        pipeline_name=_get_pipeline_name(model),
        steps=[compressor_step, optimizer_step],
    )
    return pipeline
