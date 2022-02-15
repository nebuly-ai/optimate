from pathlib import Path

import torch

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import NVIDIA_FILENAMES
from nebullvm.inference_learners.tensor_rt import (
    NVIDIA_INFERENCE_LEARNERS,
    NvidiaInferenceLearner,
)
from nebullvm.optimizers.base import BaseOptimizer

if torch.cuda.is_available():
    import tensorrt as trt


class TensorRTOptimizer(BaseOptimizer):
    def _build_and_save_the_engine(
        self, engine_path: str, onnx_model_path: str
    ):
        # -- Build phase --
        nvidia_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(nvidia_logger)
        # create network definition
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        # import the model
        parser = trt.OnnxParser(network, nvidia_logger)
        success = parser.parse_from_file(onnx_model_path)

        if not success:
            for idx in range(parser.num_errors):
                if self.logger is not None:
                    self.logger.debug(parser.get_error(idx))
            raise ValueError(
                f"Errors occurred while processing the "
                f"ONNX file at {onnx_model_path}"
            )

        # build the engine
        # TODO: setup config value for the class in a config file
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 20  # 1 MiB (put 30 for 1GB)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> NvidiaInferenceLearner:
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )
        engine_path = Path(onnx_model).parent / NVIDIA_FILENAMES["engine"]
        self._build_and_save_the_engine(engine_path, onnx_model)
        # TODO: generalize the input/output names and allow multiple
        #  inputs / outputs.
        model = NVIDIA_INFERENCE_LEARNERS[output_library].from_engine_path(
            network_parameters=model_params,
            engine_path=engine_path,
            input_name="input",
            output_name="output",
        )
        return model
