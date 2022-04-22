import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union, Dict

import tensorflow as tf

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    InputInfo,
    ModelCompiler,
    QuantizationType,
)
from nebullvm.converters import ONNXConverter
from nebullvm.optimizers import BaseOptimizer
from nebullvm.quantizers.onnx_quantizer import ONNXQuantizerManager
from nebullvm.utils.tf import get_outputs_sizes_tf, create_model_inputs_tf
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def optimize_tf_model(
    model: Union[tf.Module, tf.keras.Model],
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    save_dir: str,
    input_types: List[str] = None,
    extra_input_info: List[Dict] = None,
    dynamic_axis: Dict = None,
    quantization_ths: float = None,
    ignore_compilers: List[str] = None,
    custom_optimizers: List[BaseOptimizer] = None,
):
    """Basic function for optimizing a tensorflow model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.

    Args:
        model (tf.Module or keras.Model): Model that needs optimization.
        batch_size (int): The model batch size. Note that nebullvm does not
            support at the moment dynamic batch size, so a valid input should
            be given.
        input_sizes (List[Tuple]]): List containing the size of all the input
            tensors of the model. Note that even just a single tensor is needed
            as model input, this field must be a list containing (in the
            exposed case) a single element). The tuple must contain all the
            input tensor dimensions excluding the batch size. This means that
            the final input tensor size will be considered as
            `(batch_size, *input_tensor_size)`, where `input_tensor_size` is
            one list element of `input_sizes`.
        save_dir (str): Path to the directory where saving the final model.
        input_types (List[str], optional): List of input types. If no value is
            given all the inputs will be considered as float type. The
            supported string values are "int" and "float".
        extra_input_info (List[Dict], optional): List of extra information
            needed for defining the input tensors, e.g. max_value and min_value
            the tensors can get.
        dynamic_axis (Dict, optional): Dictionary containing info about the
            dynamic axis. It should contain as keys both "inputs" and "outputs"
            and as values two lists of dictionaries where each dictionary
            represents the dynamic axis information for an input/output tensor.
            The inner dictionary should have as key an integer, i.e. the
            dynamic axis (considering also the batch size) and as value a
            string giving a "tag" to it, e.g. "batch_size".
        quantization_ths (float, optional): Tolerated relative error for
            performing quantization before compiling the model. If no value
            is given, no quantization will be performed. Note that
            just dynamic quantization will be performed, since no
            data is given as input. For using other types of quantization
            please use `optimize_tf_model_from_data` instead.
        ignore_compilers (List[str]): List of DL compilers we want to ignore
            while running the optimization. Compiler name should be one
            between "tvm", "tensor RT", "openvino" and "onnxruntime".
        custom_optimizers (List[BaseOptimizer], optional): List of optimizers
            which can be used for producing InferenceLearners, i.e. models
            optimized for inference. This list is useful when some compilers,
            not included in the base version of nebullvm or not used
            by default, which are specific for a particular tasks, need
            to be used.

    Returns:
        BaseInferenceLearner: Optimized model usable with the classical
            tensorflow interface. Note that as a torch model it takes as input
            and it gives as output `tf.Tensor`s.
    """
    if input_types is None:
        input_types = ["float"] * len(input_sizes)
    if extra_input_info is None:
        extra_input_info = [{}] * len(input_sizes)
    if not len(input_sizes) == len(input_types) == len(extra_input_info):
        raise ValueError(
            f"Mismatch in the input list lengths. Given {len(input_sizes)} "
            f"sizes, {len(input_types)} input types and "
            f"{len(extra_input_info)} extra input infos."
        )
    input_infos = [
        InputInfo(size=input_size, dtype=input_type, **extra_info)
        for input_size, input_type, extra_info in zip(
            input_sizes, input_types, extra_input_info
        )
    ]
    dl_library = DeepLearningFramework.TENSORFLOW
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=input_infos,
        output_sizes=get_outputs_sizes_tf(
            model,
            input_tensors=create_model_inputs_tf(batch_size, input_infos),
        ),
        dynamic_info=dynamic_axis,
    )
    ignore_compilers = (
        []
        if ignore_compilers is None
        else [ModelCompiler(compiler) for compiler in ignore_compilers]
    )
    model_converter = ONNXConverter()
    model_optimizer = MultiCompilerOptimizer(
        ignore_compilers=ignore_compilers,
        extra_optimizers=custom_optimizers,
        debug_mode=int(os.environ.get("DEBUG_MODE", "0")) > 0,
    )
    with TemporaryDirectory() as tmp_dir:
        onnx_path = model_converter.convert(
            model, model_params.input_sizes, Path(tmp_dir)
        )
        if quantization_ths is not None:
            quantization_manager = ONNXQuantizerManager(quantization_ths)
            quantized_onnx_path = quantization_manager.run(
                str(onnx_path),
                model_params,
                quantization_type=QuantizationType.DYNAMIC,
            )
            if quantized_onnx_path is not None:
                onnx_path = Path(quantized_onnx_path)
        model_optimized = model_optimizer.optimize(
            str(onnx_path), dl_library, model_params
        )
        model_optimized.save(save_dir)
    return model_optimized.load(save_dir)
