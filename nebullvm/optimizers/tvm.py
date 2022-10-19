import logging
import os
import uuid
from typing import Tuple, Dict, Optional, Callable, Any

import onnx
import torch.cuda

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.config import (
    AUTO_TVM_TUNING_OPTION,
    AUTO_TVM_PARAMS,
    QUANTIZATION_DATA_NUM,
    CONSTRAINED_METRIC_DROP_THS,
)
from nebullvm.inference_learners.tvm import (
    TVM_INFERENCE_LEARNERS,
    ApacheTVMInferenceLearner,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers.base import BaseOptimizer
from nebullvm.optimizers.quantization.tvm import TVMCalibrator
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    get_input_names,
)
from nebullvm.utils.torch import create_model_inputs_torch

try:
    import tvm
    from tvm import IRModule
    from tvm.runtime.ndarray import NDArray
    from tvm.autotvm.tuner import XGBTuner
    from tvm import autotvm
    import tvm.relay as relay
    from tvm.relay.transform import ToMixedPrecision
except ImportError:
    # TVM is installed in the inference_learner package.
    # TVM objects needed for avoiding errors:
    IRModule = object
    NDArray = object


class ApacheTVMOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Nvidia GPUs using TensorRT."""

    def optimize_from_torch(
        self,
        torch_model: torch.nn.Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[ApacheTVMInferenceLearner]:
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        target = self._get_target()
        mod, params = self._build_tvm_model_from_torch(
            torch_model, model_params
        )
        if quantization_type is not None:
            if quantization_type is QuantizationType.HALF:
                mod = ToMixedPrecision(mixed_precision_type="float16")(mod)
            else:
                if quantization_type is QuantizationType.DYNAMIC:
                    inputs = None
                elif quantization_type is QuantizationType.STATIC:
                    inputs = input_data.get_split("train").get_numpy_list(
                        QUANTIZATION_DATA_NUM
                    )
                    input_names = [f"input_{n}" for n in range(len(inputs[0]))]
                    inputs = TVMCalibrator(inputs, input_names)
                else:
                    return
                mod = self._quantize(mod, params, input_data=inputs)
        tuning_records = self._tune_tvm_model(target, mod, params)
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=target, params=params)

        # Remove temporary file created by tvm
        os.remove(tuning_records)

        learner = TVM_INFERENCE_LEARNERS[
            DeepLearningFramework.PYTORCH
        ].from_runtime_module(
            input_tfms=input_tfms,
            network_parameters=model_params,
            lib=lib,
            target_device=target,
            input_names=[
                f"input_{i}" for i in range(len(model_params.input_infos))
            ],
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )

        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )
        is_valid = check_precision(
            learner,
            test_input_data,
            model_outputs,
            metric_drop_ths
            if quantization_type is not None
            else CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )
        if not is_valid:
            if quantization_type is None:
                self._log(
                    "The model optimized with Pytorch tvm gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner

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
        model_outputs: Any = None,
    ) -> Optional[ApacheTVMInferenceLearner]:
        """Optimize the input model with Apache TVM.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
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
            model_outputs (Any): Outputs computed by the original model.

        Returns:
            ApacheTVMInferenceLearner: Model optimized with TVM. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        check_quantization(quantization_type, metric_drop_ths)
        target = self._get_target()
        mod, params = self._build_tvm_model_from_onnx(model, model_params)
        if quantization_type is not None:
            if quantization_type is QuantizationType.HALF:
                mod = ToMixedPrecision(mixed_precision_type="float16")(mod)
            else:
                if quantization_type is QuantizationType.DYNAMIC:
                    inputs = None
                elif quantization_type is QuantizationType.STATIC:
                    inputs = input_data.get_split("train").get_numpy_list(
                        QUANTIZATION_DATA_NUM
                    )
                    inputs = TVMCalibrator(inputs, get_input_names(model))
                else:
                    return
                mod = self._quantize(mod, params, input_data=inputs)
        tuning_records = self._tune_tvm_model(target, mod, params)
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=target, params=params)

        # Remove temporary file created by tvm
        os.remove(tuning_records)

        learner = TVM_INFERENCE_LEARNERS[output_library].from_runtime_module(
            input_tfms=input_tfms,
            network_parameters=model_params,
            lib=lib,
            target_device=target,
            input_names=get_input_names(model),
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )
        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )
        is_valid = check_precision(
            learner,
            test_input_data,
            model_outputs,
            metric_drop_ths
            if quantization_type is not None
            else CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )
        if not is_valid:
            if quantization_type is None:
                self._log(
                    "The model optimized with ONNX tvm gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner

    @staticmethod
    def _build_tvm_model_from_torch(
        torch_model: torch.nn.Module, model_params: ModelParams
    ) -> Tuple[IRModule, Dict[str, NDArray]]:
        shape_dict = {
            f"input_{i}": (
                model_params.batch_size,
                *input_size,
            )
            for i, input_size in enumerate(model_params.input_sizes)
        }
        inputs = tuple(
            create_model_inputs_torch(
                model_params.batch_size, model_params.input_infos
            )
        )
        if torch.cuda.is_available():
            inputs = tuple(input_.cpu() for input_ in inputs)
            torch_model.cpu()
        with torch.no_grad():
            _ = torch_model(*inputs)
            model_trace = torch.jit.trace(torch_model, inputs)
            model_trace.eval()
        mod, params = relay.frontend.from_pytorch(
            model_trace, list(shape_dict.items())
        )
        return mod, params

    @staticmethod
    def _build_tvm_model_from_onnx(
        onnx_model_path: str, model_params: ModelParams
    ) -> Tuple[IRModule, Dict[str, NDArray]]:
        shape_dict = {
            input_key: (
                model_params.batch_size,
                *input_size,
            )
            for input_key, input_size in zip(
                get_input_names(onnx_model_path), model_params.input_sizes
            )
        }
        onnx_model = onnx.load(onnx_model_path)
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params

    @staticmethod
    def _quantize(
        mod: IRModule,
        params: Dict[str, NDArray],
        input_data: TVMCalibrator = None,
    ) -> IRModule:
        if input_data is not None:
            with relay.quantize.qconfig(
                calibrate_mode="kl_divergence", weight_scale="max"
            ):
                mod = relay.quantize.quantize(mod, params, dataset=input_data)
        else:
            with relay.quantize.qconfig(
                calibrate_mode="global_scale", global_scale=8.0
            ):
                mod = relay.quantize.quantize(mod, params)
        return mod

    @staticmethod
    def _get_target() -> str:
        force_on_cpu = int(os.getenv("TVM_ON_CPU", 0)) > 1
        if not force_on_cpu and torch.cuda.is_available():
            return str(tvm.target.cuda())
        else:
            return "llvm"  # run on CPU

    @staticmethod
    def _tune_tvm_model(
        target: str, mod: IRModule, params: Dict[str, NDArray]
    ) -> str:
        """Tune the model using AutoTVM."""
        # TODO: add support to Ansor
        tuning_records = f"{uuid.uuid4()}_model_records.json"
        # create a TVM runner
        runner = autotvm.LocalRunner(
            number=AUTO_TVM_PARAMS["number"],
            repeat=AUTO_TVM_PARAMS["repeat"],
            timeout=AUTO_TVM_PARAMS["timeout"],
            min_repeat_ms=AUTO_TVM_PARAMS["min_repeat_ms"],
            # TODO modify min_repeat_ms for GPU usage
            enable_cpu_cache_flush=True,
        )
        # begin by extracting the tasks from the onnx model
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params
        )

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(
                    AUTO_TVM_TUNING_OPTION["trials"], len(task.config_space)
                ),
                early_stopping=AUTO_TVM_TUNING_OPTION["early_stopping"],
                measure_option=autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"),
                    runner=runner,
                ),
                callbacks=[
                    autotvm.callback.log_to_file(tuning_records),
                ],
            )
        return tuning_records
