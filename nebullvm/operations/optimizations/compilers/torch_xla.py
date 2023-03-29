from typing import List, Tuple

from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.torch import (
    torch,
    Module,
)
from nebullvm.optional_modules.torch_xla import xm
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import MultiStageTransformation


class TorchXLACompiler(Compiler):
    supported_ops = {
        "cpu": [],
        "gpu": [],
        "tpu": [None]
    }

    def execute(
        self,
        model: torch.nn.Module,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model.
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

        self.compiled_model = self._compile_model(
            model, input_data, quantization_type
        )

    @torch.no_grad()
    def _compile_model(
        self,
        model: torch.nn.Module,
        input_data: DataManager,
        quantization_type: QuantizationType,
    ) -> torch.nn.Module:
        device = xm.xla_device()
        compiled_model = model.to(device)
        return compiled_model

    @torch.no_grad()
    def _quantize_model(
        self,
        model: Module,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data_torch: List[Tuple[torch.Tensor, ...]],
    ):
        raise NotImplementedError()
