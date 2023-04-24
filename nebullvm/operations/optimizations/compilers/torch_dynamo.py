from typing import Union, Any

from nebullvm.core.models import ModelParams, QuantizationType
from nebullvm.operations.optimizations.compilers.base import Compiler

from nebullvm.optional_modules.torch import (
    torch,
    Module,
    GraphModule,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class TorchDynamoCompiler(Compiler):
    supported_ops = {
        "cpu": [None],
        "gpu": [None],
    }

    def execute(
        self,
        model: Module,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
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

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        self.compiled_model = self._compile_model(model, model_params)

    @torch.no_grad()
    def _compile_model(
        self,
        model: Union[Module, GraphModule],
        network_parameters: ModelParams,
    ) -> Any:
        dynamic = False
        if network_parameters.dynamic_info is not None:
            dynamic = True
        return torch.compile(model, dynamic=dynamic)

    def _quantize_model(self, **kwargs) -> Any:
        raise NotImplementedError
