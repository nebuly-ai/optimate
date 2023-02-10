import abc
import os
import uuid
from abc import ABC
from typing import Any, Tuple, Dict, Union

from nebullvm.config import (
    AUTO_TVM_PARAMS,
    AUTO_TVM_TUNING_OPTION,
)
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.quantizations.tvm import (
    TVMCalibrator,
    quantize_apache_tvm,
)
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.onnx import onnx
from nebullvm.optional_modules.torch import Module, torch
from nebullvm.optional_modules.tvm import (
    tvm,
    IRModule,
    NDArray,
    XGBTuner,
    autotvm,
    relay,
    ExecutorFactoryModule,
)
from nebullvm.tools.base import (
    QuantizationType,
    ModelParams,
    Device,
    DeviceType,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import get_input_names
from nebullvm.tools.pytorch import create_model_inputs_torch
from nebullvm.tools.transformations import MultiStageTransformation


class ApacheTVMCompiler(Compiler, ABC):
    supported_ops = {
        "cpu": [
            None,
            # QuantizationType.STATIC,
            QuantizationType.HALF,
            QuantizationType.DYNAMIC,
        ],
        "gpu": [
            None,
            # QuantizationType.STATIC,
            QuantizationType.HALF,
            QuantizationType.DYNAMIC,
        ],
    }

    def __init__(self):
        super().__init__()
        self.model_orig = None

    def execute(
        self,
        model: Union[Module, str],
        input_tfms: MultiStageTransformation,
        model_params: ModelParams,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Compile the input model using Apache TVM compiler.

        Args:
            model (Union[Module, str]: The input model. Can be a torch model
                or a path to an onnx model.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            model_params (ModelParams): Model parameters.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with a higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            input_data (DataManager): User defined data. Default: None
        """

        if quantization_type not in self.supported_ops[self.device.type.value]:
            self.compiled_model = None
            return

        if quantization_type is QuantizationType.STATIC and input_data is None:
            raise ValueError("Input data is required for static quantization.")

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        check_quantization(quantization_type, metric_drop_ths)

        mod, params = self._build_tvm_model(model, model_params)

        if quantization_type is not None:
            mod = self._quantize_model(
                mod, quantization_type, input_tfms, input_data, params
            )

        self.compiled_model = self._compile_model(mod, params)

    @abc.abstractmethod
    def _build_tvm_model(self, model: Any, model_params: ModelParams):
        raise NotImplementedError()

    @staticmethod
    def _build_tvm_model_from_torch(
        torch_model: Module, model_params: ModelParams, device: Device
    ) -> Tuple[IRModule, Dict[str, NDArray]]:
        shape_dict = {
            f"input_{i}": input_size
            for i, input_size in enumerate(model_params.input_sizes)
        }
        inputs = tuple(create_model_inputs_torch(model_params.input_infos))
        if device.type is not DeviceType.GPU:
            inputs = tuple(input_.cpu() for input_ in inputs)
            torch_model.cpu()
        else:
            inputs = tuple(
                input_.to(device.to_torch_format()) for input_ in inputs
            )
            torch_model.to(device.to_torch_format())
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
            input_key: input_size
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
    def _get_target(device) -> str:
        if device.type is DeviceType.GPU:
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

    def _compile_model(self, model: Any, params: Any) -> ExecutorFactoryModule:
        target = self._get_target(self.device)
        tuning_records = self._tune_tvm_model(target, model, params)
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(model, target=target, params=params)

        # Remove temporary file created by tvm
        os.remove(tuning_records)

        return lib

    @staticmethod
    def _quantize_model(
        model: Any,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data: DataManager,
        params,
    ):
        return quantize_apache_tvm(
            model, quantization_type, input_tfms, input_data, params
        )


class PyTorchApacheTVMCompiler(ApacheTVMCompiler):
    def _build_tvm_model(self, model: Any, model_params: ModelParams):
        return self._build_tvm_model_from_torch(
            model, model_params, self.device
        )


class ONNXApacheTVMCompiler(ApacheTVMCompiler):
    def _build_tvm_model(self, model: Any, model_params: ModelParams):
        self.model_orig = model
        return self._build_tvm_model_from_onnx(model, model_params)
