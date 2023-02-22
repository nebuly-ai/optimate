from pathlib import Path
from typing import Union

from nebullvm.operations.conversions.converters import (
    PytorchConverter,
)
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.optional_modules.torch import (
    Module,
    GraphModule,
)
from nebullvm.tools.base import (
    ModelParams,
    QuantizationType,
)
from nebullvm.tools.data import DataManager


class DeepSparseCompiler(Compiler):
    supported_ops = {
        "cpu": [None],
        "gpu": [],
    }

    def __init__(self):
        super().__init__()
        self.conversion_op = PytorchConverter()

    def execute(
        self,
        model: Module,
        onnx_output_path: str,
        model_params: ModelParams,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Compile the input model using DeepSparse Compiler.

        Args:
            model (torch.nn.Module): The pytorch model.
            onnx_output_path (str): Path where the converted ONNX model will be
                stored.
            model_params (ModelParams): The model parameters.
            quantization_type (QuantizationType): The desired
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

        self.compiled_model = self._compile_model(
            model, onnx_output_path, input_data, model_params
        )

    def _compile_model(
        self,
        model: Union[Module, GraphModule],
        onnx_output_path: str,
        input_data: DataManager,
        model_params: ModelParams,
    ) -> str:
        self.conversion_op.model_name = "model_pruned"
        onnx_pruned_path = Path(onnx_output_path)
        self.conversion_op.to(self.device).set_state(
            model, input_data
        ).execute(onnx_pruned_path, model_params)
        onnx_pruned_path = str(onnx_pruned_path / "model_pruned.onnx")
        return onnx_pruned_path

    @staticmethod
    def _quantize_model(**kwargs):
        raise NotImplementedError()
