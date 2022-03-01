import os
import uuid
from typing import Tuple, Dict

import onnx
import torch.cuda

from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.config import (
    AUTO_TVM_TUNING_OPTION,
    AUTO_TVM_PARAMS,
    NO_COMPILER_INSTALLATION,
)
from nebullvm.inference_learners.tvm import (
    TVM_INFERENCE_LEARNERS,
    ApacheTVMInferenceLearner,
)
from nebullvm.optimizers.base import BaseOptimizer, get_input_names

try:
    import tvm
    from tvm import IRModule
    from tvm.runtime.ndarray import NDArray
    from tvm.autotvm.tuner import XGBTuner
    from tvm import autotvm
    import tvm.relay as relay
except ImportError:
    import warnings

    if not NO_COMPILER_INSTALLATION:
        warnings.warn(
            "Not found any valid tvm installation. "
            "Trying to install it from source."
        )
        from nebullvm.installers.installers import install_tvm

        install_tvm()
        import tvm
        from tvm import IRModule
        from tvm.runtime.ndarray import NDArray
        from tvm.autotvm.tuner import XGBTuner
        from tvm import autotvm
        import tvm.relay as relay
    else:
        warnings.warn("Not found any valid tvm installation")
        # TVM objects needed for avoiding errors
        IRModule = object
        NDArray = object


class ApacheTVMOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Nvidia GPUs using TensorRT."""

    def optimize_from_torch(
        self,
        torch_model: torch.nn.Module,
        model_params: ModelParams,
    ):
        target = self._get_target()
        mod, params = self._build_tvm_model_from_torch(
            torch_model, model_params
        )
        tuning_records = self._tune_tvm_model(target, mod, params)
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=target, params=params)
        model = TVM_INFERENCE_LEARNERS[
            DeepLearningFramework.PYTORCH
        ].from_runtime_module(
            network_parameters=model_params,
            lib=lib,
            target_device=target,
            input_names=[
                f"input_{i}" for i in range(len(model_params.input_sizes))
            ],
        )
        return model

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> ApacheTVMInferenceLearner:
        """Optimize the input model with Apache TVM.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.

        Returns:
            ApacheTVMInferenceLearner: Model optimized with TVM. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        target = self._get_target()
        mod, params = self._build_tvm_model_from_onnx(onnx_model, model_params)
        tuning_records = self._tune_tvm_model(target, mod, params)
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=target, params=params)
        model = TVM_INFERENCE_LEARNERS[output_library].from_runtime_module(
            network_parameters=model_params,
            lib=lib,
            target_device=target,
            input_names=get_input_names(onnx_model),
        )
        return model

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
            torch.randn(input_shape) for input_shape in shape_dict.values()
        )
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
