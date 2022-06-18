import os
import subprocess
from pathlib import Path

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.inference_learners.onnx_mlir import (
    ONNX_MLIR_INFERENCE_LEARNERS,
    ONNXMlirInferenceLearner,
)
from nebullvm.optimizers.base import BaseOptimizer


class ONNXMlirOptimizer(BaseOptimizer):
    """Class for compiling the AI models from ONNX format to equivalent MLIR dialect."""

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> ONNXMlirInferenceLearner:
        """Optimize the onnx model to MLIR Compiler Infrastructure.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.

        Returns:
            ONNXMlirInferenceLearner: Model optimized with ONNX-MLIR. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        inputs = list(model_params.input_sizes)

        shape_info = "--shapeInformation="
        for input_index in range(len(inputs)):
            shape_info += (
                f"{input_index}:{model_params.batch_size}x"
                + f"x".join(map(str, inputs[input_index]))
                + ","
            )
        shape_info = shape_info[:-1]

        command = [
            "./onnx-mlir",
            "--EmitLib",
            "--O3",
            shape_info,
            onnx_model,
        ]
        process = subprocess.Popen(
            command,
            cwd=os.path.join(
                os.environ.get("ONNX_MLIR_HOME", ""),
                "bin",
            ),
        )
        process.wait()

        base_path = Path(onnx_model).parent
        onnx_mlir_model_path = base_path / f"{Path(onnx_model).stem}.so"

        model = ONNX_MLIR_INFERENCE_LEARNERS[output_library](
            onnx_mlir_model_path=str(onnx_mlir_model_path),
            network_parameters=model_params,
        )

        return model
