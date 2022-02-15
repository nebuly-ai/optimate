from pathlib import Path
import subprocess

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.inference_learners.openvino import (
    OPENVINO_INFERENCE_LEARNERS,
    OpenVinoInferenceLearner,
)
from nebullvm.optimizers.base import BaseOptimizer


class OpenVinoOptimizer(BaseOptimizer):
    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> OpenVinoInferenceLearner:
        process = subprocess.Popen(
            [
                "mo",
                "--input_model",
                onnx_model,
                "--output_dir",
                str(Path(onnx_model).parent),
            ],
        )
        process.wait()
        base_path = Path(onnx_model).parent
        openvino_model_path = base_path / f"{Path(onnx_model).stem}.xml"
        openvino_model_weights = base_path / f"{Path(onnx_model).stem}.bin"
        model = OPENVINO_INFERENCE_LEARNERS[output_library].from_model_name(
            model_name=str(openvino_model_path),
            model_weights=str(openvino_model_weights),
            network_parameters=model_params,
        )
        return model
