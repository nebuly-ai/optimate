import subprocess
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np

from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.compilers.quantizations.openvino import (  # noqa: E501
    quantize_openvino,
)
from nebullvm.operations.optimizations.compilers.quantizations.utils import (
    check_quantization,
)
from nebullvm.optional_modules.openvino import (
    Core,
    CompiledModel,
)
from nebullvm.tools.base import (
    QuantizationType,
    ModelParams,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import get_input_names
from nebullvm.tools.transformations import MultiStageTransformation


class OpenVINOCompiler(Compiler):
    supported_ops = {
        "cpu": [
            None,
            QuantizationType.STATIC,
            QuantizationType.HALF,
        ],
        "gpu": [],
    }

    def __init__(self):
        super().__init__()

    def execute(
        self,
        model: Union[str, Path],
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Compile the input model using OpenVINO library.

        Args:
            model (str): The onnx model path.
            model_params (ModelParams): The model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
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
        train_input_data = input_data.get_split("train").get_numpy_list(
            QUANTIZATION_DATA_NUM
        )

        cmd = [
            "mo",
            "--input_model",
            str(model),
            "--output_dir",
            str(Path(model).parent),
            "--input",
            ",".join(get_input_names(str(model))),
            "--input_shape",
            ",".join([f"{list(shape)}" for shape in model_params.input_sizes]),
        ]

        if quantization_type is QuantizationType.DYNAMIC:
            return None

        if quantization_type is QuantizationType.HALF:
            cmd = cmd + ["--compress_to_fp16"]

        process = subprocess.Popen(cmd)
        process.wait()
        base_path = Path(model).parent
        openvino_model_path = base_path / f"{Path(model).stem}.xml"
        openvino_model_weights = base_path / f"{Path(model).stem}.bin"

        if quantization_type not in [QuantizationType.HALF, None]:
            openvino_model_path, openvino_model_weights = self._quantize_model(
                model_topology=str(openvino_model_path),
                model_weights=str(openvino_model_weights),
                input_names=get_input_names(str(model)),
                input_data=train_input_data,
            )

        self.compiled_model = str(
            Path(openvino_model_path).parent / Path(openvino_model_path).stem
        )

    def _compile_model(
        self,
        model_name: str,
        model_weights: str,
        network_parameters: ModelParams,
    ) -> CompiledModel:
        core = Core()
        model = core.read_model(model=model_name, weights=model_weights)

        dynamic_shape = self._get_dynamic_shape(model, network_parameters)

        if dynamic_shape is not None:
            model.reshape(dynamic_shape)

        return core.compile_model(model=model, device_name="CPU")

    @staticmethod
    def _quantize_model(
        model_topology: str,
        model_weights: str,
        input_data: List[Tuple[np.ndarray, ...]],
        input_names: List[str],
    ) -> Tuple[str, str]:
        return quantize_openvino(
            model_topology, model_weights, input_data, input_names
        )
