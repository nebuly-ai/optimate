import logging
from pathlib import Path
import subprocess
from typing import Optional, Callable, Any, Tuple

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    QuantizationType,
    Device,
)
from nebullvm.config import QUANTIZATION_DATA_NUM, CONSTRAINED_METRIC_DROP_THS
from nebullvm.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
    OpenVinoInferenceLearner,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers.base import BaseOptimizer
from nebullvm.optimizers.quantization.openvino import quantize_openvino
from nebullvm.optimizers.quantization.utils import check_precision
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    get_input_names,
)

logger = logging.getLogger("nebullvm_logger")


class OpenVinoOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Intel CPUs using OpenVino."""

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        device: Device,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
        model_outputs: Any = None,
    ) -> Optional[Tuple[OpenVinoInferenceLearner, float]]:
        """Optimize the onnx model with OpenVino.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            device: (Device): Device where the model will be run.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction. Default: None.
            input_data (DataManager, optional): User defined data.
                Default: None.
            model_outputs (Any, optional): Outputs computed by the original
                model. Default: None.

        Returns:
            OpenVinoInferenceLearner: Model optimized with OpenVino. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        logger.info(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
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

        train_input_data = input_data.get_split("train").get_numpy_list(
            QUANTIZATION_DATA_NUM
        )

        if quantization_type not in [QuantizationType.HALF, None]:
            # Add post training optimization
            (openvino_model_path, openvino_model_weights,) = quantize_openvino(
                model_topology=str(openvino_model_path),
                model_weights=str(openvino_model_weights),
                input_names=get_input_names(model),
                input_data=train_input_data,
            )

        learner = OPENVINO_INFERENCE_LEARNERS[output_library].from_model_name(
            model_name=str(openvino_model_path),
            model_weights=str(openvino_model_weights),
            network_parameters=model_params,
            input_tfms=input_tfms,
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )

        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )

        is_valid, metric_drop = check_precision(
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
                logger.warning(
                    "The model optimized with openvino gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped."
                )
            return None
        return learner, metric_drop
