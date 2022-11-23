import os
from pathlib import Path
import subprocess
from typing import Union, Any, List

from nebullvm.base import (
    ModelParams,
    QuantizationType,
    DeepLearningFramework,
)
from nebullvm.config import QUANTIZATION_DATA_NUM, TORCH_TENSORRT_PRECISIONS
from nebullvm.operations.optimizations.quantizations.tensor_rt import (
    ONNXTensorRTQuantizer,
)
from nebullvm.operations.optimizations.compilers.base import Compiler
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
)
from nebullvm.optional_modules.tensor_rt import tensorrt as trt
from nebullvm.optional_modules.torch import torch, Module
from nebullvm.optional_modules.torch_tensorrt import (
    torch_tensorrt,
    DataLoaderCalibrator,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.transformations.precision_tfms import HalfPrecisionTransformation
from nebullvm.utils.data import DataManager, PytorchDataset
from nebullvm.utils.onnx import get_input_names


class TensorRTCompiler(Compiler):
    supported_ops = {
        "cpu": [],
        "gpu": [
            None,
            QuantizationType.STATIC,
            QuantizationType.HALF,
        ],
    }

    def __init__(self, dl_framework: DeepLearningFramework):
        super().__init__()
        self.dl_framework = dl_framework
        self.onnx_model = None

    def execute(self, *args, **kwargs):
        if self.dl_framework is DeepLearningFramework.PYTORCH:
            compile_op = PyTorchTensorRTCompiler()
        elif self.dl_framework is DeepLearningFramework.NUMPY:
            compile_op = ONNXTensorRTCompiler()
        else:
            raise ValueError(
                f"TensorRT is not supported for {self.dl_framework} models."
            )

        compile_op.to(self.device).execute(*args, **kwargs)

        self.compiled_model = compile_op.compiled_model
        self.onnx_model = (
            compile_op.onnx_model
            if hasattr(compile_op, "onnx_model")
            else None
        )

    def compile_model(self, **kwargs) -> Any:
        pass


class PyTorchTensorRTCompiler(TensorRTCompiler):
    def __init__(self):
        super().__init__(DeepLearningFramework.PYTORCH)

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

        if quantization_type is QuantizationType.HALF:
            dtype = torch.half
            input_tfms.append(HalfPrecisionTransformation())
        elif quantization_type is QuantizationType.STATIC:
            dtype = torch.int8

            dataset = PytorchDataset(input_data.get_split("train"))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                num_workers=0,
            )

            calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
                dataloader,
                use_cache=False,
                algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,  # noqa E501
                device=torch.device("cuda:0"),
            )
        else:
            dtype = torch.float32

        # Convert int64 to int32 for transformers inputs
        input_tensors = [
            tensor.cuda()
            if tensor.dtype != torch.int64
            else tensor.to(torch.int32).cuda()
            for tensor in input_data.get_list(1)[0]
        ]

        self.compiled_model = self.compile_model(
            model=model,
            input_tensors=input_tensors,
            dtype=dtype,
            calibrator=calibrator
            if quantization_type is QuantizationType.STATIC
            else None,  # noqa E501
            quantization_type=quantization_type,
        )

    def compile_model(
        self,
        model: Module,
        input_tensors: List[torch.Tensor],
        dtype: torch.dtype,
        calibrator: DataLoaderCalibrator,
        quantization_type: QuantizationType,
    ):

        model.cuda().eval()

        try:
            torch.jit.script(model)
        except Exception:
            model = torch.jit.trace(model, input_tensors)

        with torch_tensorrt.logging.errors():
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        tensor.shape,
                        dtype=torch.half
                        if (
                            dtype == torch.half
                            and tensor.dtype not in [torch.int8, torch.int32]
                        )
                        else tensor.dtype,
                    )
                    for tensor in input_tensors
                ],
                enabled_precisions=TORCH_TENSORRT_PRECISIONS[str(dtype)],
                calibrator=calibrator
                if quantization_type is QuantizationType.STATIC
                else None,
                workspace_size=1 << 30,
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                    "dla_core": 0,
                    "allow_gpu_fallback": False,
                    "disable_tf32": False,
                },
                truncate_long_and_double=True,
            )

        # Delete calibration cache
        if os.path.exists("calibration.cache"):
            os.remove("calibration.cache")

        return trt_model


class ONNXTensorRTCompiler(TensorRTCompiler):
    def __init__(self):
        super().__init__(DeepLearningFramework.NUMPY)
        self.onnx_model = None
        self.quantization_op = ONNXTensorRTQuantizer()

    def execute(
        self,
        model: Union[str, Path],
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

        try:
            import onnxsim  # noqa: F401

            # Simplify model, otherwise tensor RT won't work on gpt2 and some
            # other models.
            simplified_model = str(model) + "_simplified"
            if not Path(simplified_model).is_file():
                cmd = [
                    "onnxsim",
                    str(model),
                    simplified_model,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL)

            # First try with simplified model
            onnx_model_path = simplified_model
        except Exception:
            # Try again with original model
            onnx_model_path = str(model)

        # -- Build phase --
        nvidia_logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(nvidia_logger)
        # create network definition
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        # build the engine
        # TODO: setup config value for the class in a config file
        config = builder.create_builder_config()
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        except AttributeError:
            # The method set_memory_pool_limit is not available
            # until TensorRT Release 8.4.1
            self.logger.warning(
                "Cannot call method set_memory_pool_limit for TensorRT."
                "Please update TensorRT version."
            )

        if quantization_type is not None:
            self.quantization_op.to(self.device).execute(
                quantization_type,
                model_params,
                config,
                input_tfms,
                train_input_data
                if quantization_type is QuantizationType.STATIC
                else None,
            )
            config = self.quantization_op.get_result()

        if (
            quantization_type is None
            or self.quantization_op.get_result() is not None
        ):
            self.compiled_model = self.compile_model(
                onnx_model_path=str(onnx_model_path),
                model_params=model_params,
                config=config,
                network=network,
                builder=builder,
                nvidia_logger=nvidia_logger,
            )
            self.onnx_model = onnx_model_path

    def compile_model(
        self,
        onnx_model_path: str,
        model_params: ModelParams,
        config,
        network,
        builder,
        nvidia_logger,
    ):
        parser = trt.OnnxParser(network, nvidia_logger)
        success = parser.parse_from_file(onnx_model_path)

        if not success:
            for idx in range(parser.num_errors):
                self.logger.debug(parser.get_error(idx))
            raise ValueError(
                f"Errors occurred while processing the "
                f"ONNX file at {onnx_model_path}"
            )

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
        return builder.build_serialized_network(network, config)
