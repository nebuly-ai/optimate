from pathlib import Path
import subprocess
from typing import Optional, Callable

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
    OpenVinoInferenceLearner,
)
from nebullvm.optimizers.base import BaseOptimizer
from nebullvm.optimizers.quantization.openvino import quantize_openvino
from nebullvm.optimizers.quantization.utils import check_precision
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import (
    get_input_names,
    create_model_inputs_onnx,
    run_onnx_model,
    convert_to_numpy,
)


class OpenVinoOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Intel CPUs using OpenVino."""

    def optimize(
        self,
        model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> Optional[OpenVinoInferenceLearner]:
        """Optimize the onnx model with OpenVino.

        Args:
            model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            OpenVinoInferenceLearner: Model optimized with OpenVino. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        self._log(
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
                    f"{list((model_params.batch_size,)+shape)}"
                    for shape in model_params.input_sizes
                ]
            ),
        ]
        if (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.HALF
        ):
            cmd = cmd + ["--data_type", "FP16"]
        elif (
            metric_drop_ths is not None
            and quantization_type is QuantizationType.DYNAMIC
        ):
            return None
        process = subprocess.Popen(cmd)
        process.wait()
        base_path = Path(model).parent
        openvino_model_path = base_path / f"{Path(model).stem}.xml"
        openvino_model_weights = base_path / f"{Path(model).stem}.bin"
        if (
            metric_drop_ths is not None
            and quantization_type is not QuantizationType.HALF
        ):
            if input_data is not None and quantization_type:
                input_data_onnx = input_data.get_numpy_list(300, with_ys=True)
            else:
                input_data_onnx = [
                    (
                        create_model_inputs_onnx(
                            model_params.batch_size, model_params.input_infos
                        ),
                        0,
                    )
                ]
            # Add post training optimization
            openvino_model_path, openvino_model_weights = quantize_openvino(
                model_topology=str(openvino_model_path),
                model_weights=str(openvino_model_weights),
                input_names=get_input_names(model),
                input_data=input_data_onnx,
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
        if metric_drop_ths is not None:
            if input_data is None:
                inputs = [learner.get_inputs_example()]
                ys = None
            else:
                inputs, ys = input_data.get_list(
                    100, with_ys=True, shuffle=True
                )
            output_data_onnx = [
                tuple(
                    run_onnx_model(
                        model,
                        [convert_to_numpy(x) for x in tuple_],
                    )
                )
                for tuple_ in inputs
            ]
            is_valid = check_precision(
                learner,
                inputs,
                output_data_onnx,
                metric_drop_ths,
                metric_func=metric,
                ys=ys,
            )
            if not is_valid:
                return None
        return learner
