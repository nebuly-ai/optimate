import abc
from tempfile import TemporaryDirectory
from typing import List, Callable, Union, Tuple, Any, Dict, Type

from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.operations.base import Operation
from nebullvm.operations.inference_learners.base import BuildInferenceLearner
from nebullvm.operations.inference_learners.builders import (
    PytorchBuildInferenceLearner,
    DeepSparseBuildInferenceLearner,
    IntelNeuralCompressorBuildInferenceLearner,
    PyTorchTensorRTBuildInferenceLearner,
    ONNXTensorRTBuildInferenceLearner,
    PyTorchApacheTVMBuildInferenceLearner,
    ONNXApacheTVMBuildInferenceLearner,
    ONNXBuildInferenceLearner,
    OpenVINOBuildInferenceLearner,
    TFLiteBuildInferenceLearner,
    TensorflowBuildInferenceLearner,
)
from nebullvm.operations.measures.measures import MetricDropMeasure
from nebullvm.operations.measures.utils import (
    compute_relative_difference,
    compute_optimized_running_time,
)
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.deepsparse import (
    DeepSparseCompiler,
)
from nebullvm.operations.optimizations.compilers.intel_neural_compressor import (  # noqa: E501
    IntelNeuralCompressorCompiler,
)
from nebullvm.operations.optimizations.compilers.onnxruntime import (
    ONNXCompiler,
)
from nebullvm.operations.optimizations.compilers.openvino import (
    OpenVINOCompiler,
)
from nebullvm.operations.optimizations.compilers.pytorch import (
    PytorchBackendCompiler,
)
from nebullvm.operations.optimizations.compilers.tensor_rt import (
    PyTorchTensorRTCompiler,
    ONNXTensorRTCompiler,
)
from nebullvm.operations.optimizations.compilers.tensorflow import (
    TFLiteBackendCompiler,
    TensorflowBackendCompiler,
)
from nebullvm.operations.optimizations.compilers.tvm import (
    PyTorchApacheTVMCompiler,
    ONNXApacheTVMCompiler,
)

from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import (
    ModelCompiler,
    QuantizationType,
    OptimizationTime,
    ModelParams,
    DeepLearningFramework,
    ModelCompressor,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class Optimizer(Operation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimized_models = []
        self.source_dl_framework = None
        self.pipeline_dl_framework = None
        self.compiler_ops = {}
        self.build_inference_learner_ops = {}
        self.validity_check_op = MetricDropMeasure()

    def execute(
        self,
        model: Any,
        input_data: DataManager,
        optimization_time: OptimizationTime,
        metric_drop_ths: str,
        metric: Callable,
        model_params: ModelParams,
        model_outputs: List[Tuple[Any, ...]],
        ignore_compilers: List[ModelCompiler],
        ignore_compressors: List[ModelCompressor],
        source_dl_framework: DeepLearningFramework,
    ):
        self.source_dl_framework = source_dl_framework

        # TODO: implement and select compressors from hardware

        compilers = self._select_compilers_from_hardware()
        (
            self.compiler_ops,
            self.build_inference_learner_ops,
        ) = self._load_compilers(
            ignore_compilers=ignore_compilers,
            compilers=compilers,
        )
        self._optimize(
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
        compiler_ops = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler][
                self.pipeline_dl_framework
            ]()
            for compiler in compilers
            if compiler not in ignore_compilers
            and compiler in COMPILER_TO_OPTIMIZER_MAP
        }
        build_inference_learner_ops = {
            compiler: COMPILER_TO_INFERENCE_LEARNER_MAP[compiler][
                self.pipeline_dl_framework
            ]()
            for compiler in compilers
            if compiler not in ignore_compilers
            and compiler in COMPILER_TO_OPTIMIZER_MAP
        }

        return compiler_ops, build_inference_learner_ops

    def _optimize(
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

        optimization_info = []
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
                                source_dl_framework=self.source_dl_framework,
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

                                    opt_info_dict = {
                                        "compiler": f"{self.pipeline_dl_framework.value}_{compiler.value}",  # noqa: E501
                                        "technique": q_type.value
                                        if q_type
                                        else "none",
                                        "latency": latency,
                                    }
                                    if (
                                        metric_drop_ths is not None
                                        and q_type is not None
                                    ):
                                        opt_info_dict[
                                            "metric_loss"
                                        ] = (
                                            self.validity_check_op.measure_result  # noqa: E501
                                        )
                                        opt_info_dict[
                                            "metric"
                                        ] = metric.__name__
                                    optimization_info.append(opt_info_dict)
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
                        optimization_info.append(
                            {
                                "compiler": compiler.value,
                                "technique": q_type.value
                                if q_type
                                else "none",
                                "latency": -1,
                            }
                        )
        if self.feedback_collector is not None:
            self.feedback_collector.store_info(
                key="optimizations",
                value=optimization_info,
            )

    def get_result(self) -> List:
        return self.optimized_models


COMPILER_TO_OPTIMIZER_MAP: Dict[
    ModelCompiler, Dict[DeepLearningFramework, Type[Compiler]]
] = {
    ModelCompiler.TORCHSCRIPT: {
        DeepLearningFramework.PYTORCH: PytorchBackendCompiler
    },
    ModelCompiler.DEEPSPARSE: {
        DeepLearningFramework.PYTORCH: DeepSparseCompiler
    },
    ModelCompiler.INTEL_NEURAL_COMPRESSOR: {
        DeepLearningFramework.PYTORCH: IntelNeuralCompressorCompiler
    },
    ModelCompiler.TENSOR_RT: {
        DeepLearningFramework.PYTORCH: PyTorchTensorRTCompiler,
        DeepLearningFramework.NUMPY: ONNXTensorRTCompiler,
    },
    ModelCompiler.APACHE_TVM: {
        DeepLearningFramework.PYTORCH: PyTorchApacheTVMCompiler,
        DeepLearningFramework.NUMPY: ONNXApacheTVMCompiler,
    },
    ModelCompiler.ONNX_RUNTIME: {DeepLearningFramework.NUMPY: ONNXCompiler},
    ModelCompiler.OPENVINO: {DeepLearningFramework.NUMPY: OpenVINOCompiler},
    ModelCompiler.TFLITE: {
        DeepLearningFramework.TENSORFLOW: TFLiteBackendCompiler
    },
    ModelCompiler.XLA: {
        DeepLearningFramework.TENSORFLOW: TensorflowBackendCompiler
    },
}

COMPILER_TO_INFERENCE_LEARNER_MAP: Dict[
    ModelCompiler, Dict[DeepLearningFramework, Type[BuildInferenceLearner]]
] = {
    ModelCompiler.TORCHSCRIPT: {
        DeepLearningFramework.PYTORCH: PytorchBuildInferenceLearner
    },
    ModelCompiler.DEEPSPARSE: {
        DeepLearningFramework.PYTORCH: DeepSparseBuildInferenceLearner
    },
    ModelCompiler.INTEL_NEURAL_COMPRESSOR: {
        DeepLearningFramework.PYTORCH: IntelNeuralCompressorBuildInferenceLearner  # noqa: E501
    },
    ModelCompiler.TENSOR_RT: {
        DeepLearningFramework.PYTORCH: PyTorchTensorRTBuildInferenceLearner,
        DeepLearningFramework.NUMPY: ONNXTensorRTBuildInferenceLearner,
    },
    ModelCompiler.APACHE_TVM: {
        DeepLearningFramework.PYTORCH: PyTorchApacheTVMBuildInferenceLearner,
        DeepLearningFramework.NUMPY: ONNXApacheTVMBuildInferenceLearner,
    },
    ModelCompiler.ONNX_RUNTIME: {
        DeepLearningFramework.NUMPY: ONNXBuildInferenceLearner
    },
    ModelCompiler.OPENVINO: {
        DeepLearningFramework.NUMPY: OpenVINOBuildInferenceLearner
    },
    ModelCompiler.TFLITE: {
        DeepLearningFramework.TENSORFLOW: TFLiteBuildInferenceLearner
    },
    ModelCompiler.XLA: {
        DeepLearningFramework.TENSORFLOW: TensorflowBuildInferenceLearner
    },
}
