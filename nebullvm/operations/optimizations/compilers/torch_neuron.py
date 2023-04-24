from typing import List, Tuple

from nebullvm.core.models import QuantizationType, ModelParams, DeviceType
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.torch import (
    torch,
    symbolic_trace,
)
from nebullvm.optional_modules.torch_neuron import torch_neuron
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class TorchNeuronCompiler(Compiler):
    supported_ops = {
        "cpu": [],
        "gpu": [],
        "neuron": [None, QuantizationType.HALF],
    }

    @staticmethod
    def _check_dynamic_shape(network_parameters: ModelParams) -> bool:
        """Handles case when model inputs have dynamic shapes.
        For now TorchNeuron only supports dynamic shape for the
        batch dimension.

        Args:
            network_parameters (ModelParams): The model parameters.

        Returns:
            bool: True if the model has dynamic batch size, False otherwise.
        """
        if network_parameters.dynamic_info is None:
            return False

        for i, input_shape in enumerate(
            network_parameters.dynamic_info.inputs
        ):
            if len(input_shape) > 1 or (
                len(input_shape) == 1 and input_shape.get(0) is None
            ):
                raise ValueError(
                    f"TorchNeuronCompiler only supports dynamic shapes for "
                    f"batch dimension. Provided dynamic info for input {i} "
                    f"is: {input_shape}. Please use padding for the other "
                    f"dimensions."
                )

        return True

    def execute(
        self,
        model: torch.nn.Module,
        model_params: ModelParams,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model.
            model_params (ModelParams): The model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with a higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            input_data (DataManager): User defined data. Default: None.
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
        dynamic_batch_size = self._check_dynamic_shape(model_params)

        self.compiled_model = self._compile_model(
            model,
            input_data,
            quantization_type,
            dynamic_batch_size=dynamic_batch_size,
        )

    @torch.no_grad()
    def _compile_model(
        self,
        model: torch.nn.Module,
        input_data: DataManager,
        quantization_type: QuantizationType,
        dynamic_batch_size: bool,
    ) -> torch.jit.ScriptModule:
        input_sample = input_data.get_list(1)[0]
        if self.device.type is DeviceType.GPU:
            if quantization_type is QuantizationType.HALF:
                input_sample = [
                    t.to(self.device.to_torch_format()).half()
                    if torch.is_floating_point(t)
                    else t.to(self.device.to_torch_format())
                    for t in input_sample
                ]
            else:
                input_sample = [
                    t.to(self.device.to_torch_format()) for t in input_sample
                ]
            model.to(self.device.to_torch_format())
        model.eval()

        try:
            model_scripted = symbolic_trace(model)
            model_scripted = torch_neuron.trace(
                model_scripted,
                input_sample,
                dynamic_batch_size=dynamic_batch_size,
                compiler_args=["--fast-math", "none"]
                if quantization_type is None
                else None,
            )
        except Exception:
            try:
                model_scripted = torch_neuron.trace(
                    model,
                    input_sample,
                    dynamic_batch_size=dynamic_batch_size,
                    compiler_args=["--fast-math", "none"]
                    if quantization_type is None
                    else None,
                )
            except Exception:
                raise RuntimeError("Unable to trace model with torch_neuron.")

        return model_scripted

    @torch.no_grad()
    def _quantize_model(
        self,
        model: torch.nn.Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data_torch: List[Tuple[torch.Tensor, ...]],
    ):
        raise NotImplementedError()
