from pathlib import Path
from typing import Union

from nebullvm.base import QuantizationType, ModelParams
from nebullvm.converters import ONNXConverter
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.quantizations.pytorch import (
    PytorchQuantizer,
)
from nebullvm.optional_modules.torch import (
    torch,
    Module,
    GraphModule,
)
from nebullvm.utils.data import DataManager


class DeepSparseCompiler(Compiler):
    supported_ops = {
        "cpu": [None],
        "gpu": [None],
    }

    def __init__(self):
        super().__init__()
        self.quantization_op = PytorchQuantizer()

    def execute(
        self,
        model: Module,
        input_data: DataManager,
        onnx_output_path: str,
        model_params: ModelParams,
        quantization_type: QuantizationType = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            input_data (DataManager): User defined data. Default: None.
            onnx_output_path (str): Path where the converted ONNX model will be
                stored.
            quantization_type (QuantizationType): The desired
                quantization algorithm to be used. Default: None.
            model_params (ModelParams): The model parameters.

        Returns:
            PytorchBackendInferenceLearner: Model optimized for inference.
        """

        if quantization_type not in self.supported_ops[self.device.value]:
            self.compiled_model = None
            return

        self.logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        self.compiled_model = self.compile_model(model, onnx_output_path, input_data, model_params)

    def compile_model(
        self,
        model: Union[Module, GraphModule],
        onnx_output_path: str,
        input_data: DataManager,
        model_params: ModelParams,
    ) -> str:
        converter = ONNXConverter(model_name="model_pruned")
        onnx_pruned_path = Path(onnx_output_path)
        converter.convert(
            model, model_params, onnx_pruned_path, self.device, input_data
        )
        onnx_pruned_path = str(onnx_pruned_path / "model_pruned.onnx")
        return onnx_pruned_path