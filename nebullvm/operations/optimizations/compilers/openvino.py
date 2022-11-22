from pathlib import Path
import subprocess
from typing import Tuple, Optional, Dict

from nebullvm.base import (
    ModelParams,
    QuantizationType,
)
from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.operations.optimizations.quantizations.openvino import OpenVINOQuantizer
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
)
from nebullvm.optional_modules.torch import Module
from nebullvm.optional_modules.openvino import (
    Core,
    Model,
    CompiledModel,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import get_input_names


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
        self.quantization_op = OpenVINOQuantizer()

    def execute(
        self,
        model: Module,
        input_data: DataManager,
        input_tfms: MultiStageTransformation,
        model_params: ModelParams,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        **kwargs,
    ):
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            input_data (DataManager): User defined data. Default: None.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.

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

        check_quantization(quantization_type, metric_drop_ths)
        train_input_data = input_data.get_split("train").get_numpy_list(
            QUANTIZATION_DATA_NUM
        )

        cmd = [
            "mo",
            "--input_model",
            model,
            "--output_dir",
            str(Path(model).parent),
            "--input",
            ",".join(get_input_names(model)),
            "--input_shape",
            ",".join(
                [
                    f"{list((model_params.batch_size,) + shape)}"
                    for shape in model_params.input_sizes
                ]
            ),
        ]

        if quantization_type is QuantizationType.DYNAMIC:
            return None

        if quantization_type is QuantizationType.HALF:
            cmd = cmd + ["--data_type", "FP16"]

        process = subprocess.Popen(cmd)
        process.wait()
        base_path = Path(model).parent
        openvino_model_path = base_path / f"{Path(model).stem}.xml"
        openvino_model_weights = base_path / f"{Path(model).stem}.bin"

        if quantization_type not in [QuantizationType.HALF, None]:
            self.quantization_op.execute(
                model_topology=str(openvino_model_path),
                model_weights=str(openvino_model_weights),
                input_names=get_input_names(model),
                input_data=train_input_data,
            )
            openvino_model_path = self.quantization_op.get_result()["model"]
            openvino_model_weights = self.quantization_op.get_result()["weights"]


        self.compiled_model = self.compile_model(
            model_name=str(openvino_model_path),
            model_weights=str(openvino_model_weights),
            network_parameters=model_params,
        )

    def compile_model(
        self,
        model_name: str,
        model_weights: str,
        network_parameters: ModelParams
    ) -> CompiledModel:
        core = Core()
        model = core.read_model(model=model_name, weights=model_weights)

        dynamic_shape = self._get_dynamic_shape(model, network_parameters)

        if dynamic_shape is not None:
            model.reshape(dynamic_shape)

        return core.compile_model(model=model, device_name="CPU")

    @staticmethod
    def _get_dynamic_shape(
        model: Model, network_parameters: ModelParams
    ) -> Optional[Dict[str, Tuple[int]]]:
        if network_parameters.dynamic_info is None:
            return None

        input_names = [
            list(model_input.names)[0] for model_input in model.inputs
        ]
        input_shapes = [
            (network_parameters.batch_size, *input_info.size)
            for input_info in network_parameters.input_infos
        ]
        dynamic_shapes = []

        assert len(input_shapes) == len(
            network_parameters.dynamic_info.inputs
        ), (
            f"Number of inputs defined in dynamic info "
            f"({len(input_shapes)}) is different from the one "
            f"expected from the model "
            f"({len(network_parameters.dynamic_info.inputs)})."
        )

        for input_shape, dynamic_shape_dict in zip(
            input_shapes, network_parameters.dynamic_info.inputs
        ):
            input_shape = list(input_shape)
            for key in dynamic_shape_dict.keys():
                input_shape[int(key)] = -1
            dynamic_shapes.append(tuple(input_shape))

        dynamic_shape_dict = {
            k: v for k, v in zip(input_names, dynamic_shapes)
        }
        return dynamic_shape_dict
