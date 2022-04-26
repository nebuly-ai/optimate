from pathlib import Path

import torch

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import NVIDIA_FILENAMES, NO_COMPILER_INSTALLATION
from nebullvm.inference_learners.tensor_rt import (
    NVIDIA_INFERENCE_LEARNERS,
    NvidiaInferenceLearner,
)
from nebullvm.optimizers.base import (
    BaseOptimizer,
)
from nebullvm.utils.onnx import get_input_names, get_output_names

if torch.cuda.is_available():
    try:
        import tensorrt as trt
    except ImportError:
        from nebullvm.installers.installers import install_tensor_rt
        import warnings

        if not NO_COMPILER_INSTALLATION:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "Trying to install it from source."
            )
            install_tensor_rt()
            import tensorrt as trt
        else:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "It won't be possible to use it in the following."
            )


class TensorRTOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Nvidia GPUs using TensorRT."""

    def _build_and_save_the_engine(
        self, engine_path: str, onnx_model_path: str, model_params: ModelParams
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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if model_params.dynamic_info is not None:
            profile = builder.create_optimization_profile()
            for input_name, input_dynamic_info, input_info in zip(
                get_input_names(onnx_model_path),
                model_params.dynamic_info.inputs,
                model_params.input_infos,
            ):
                profile.set_shape(
                    input_name,
                    (
                        min(model_params.batch_size, 1)
                        if 0 in input_dynamic_info
                        else model_params.batch_size,
                        *(
                            shape
                            if i + 1 not in input_dynamic_info
                            else (input_info.min_sizes or {}).get(i + 1, 1)
                            for i, shape in enumerate(input_info.size)
                        ),
                    ),
                    (model_params.batch_size, *input_info.size),
                    (model_params.batch_size, *input_info.size),
                )
            config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
    ) -> NvidiaInferenceLearner:
        """Optimize the input model with TensorRT.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.

        Returns:
            TensorRTInferenceLearner: Model optimized with TensorRT. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )
        engine_path = Path(onnx_model).parent / NVIDIA_FILENAMES["engine"]
        self._build_and_save_the_engine(engine_path, onnx_model, model_params)
        model = NVIDIA_INFERENCE_LEARNERS[output_library].from_engine_path(
            network_parameters=model_params,
            engine_path=engine_path,
            input_names=get_input_names(onnx_model),
            output_names=get_output_names(onnx_model),
        )
        return model
