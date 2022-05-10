import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union, Dict, Optional, Callable, Any

import tensorflow as tf

from nebullvm.api.frontend.utils import (
    ifnone,
    inspect_dynamic_size,
    QUANTIZATION_METRIC_MAP,
)
from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    InputInfo,
    ModelCompiler,
)
from nebullvm.converters import ONNXConverter
from nebullvm.optimizers import BaseOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.tf import (
    get_outputs_sizes_tf,
    create_model_inputs_tf,
    run_tf_model,
)
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def _extract_dynamic_axis(
    tf_model: tf.Module,
    dataset: List[Tuple[Tuple[tf.Tensor, ...], Any]],
    input_sizes: List[Tuple[int, ...]],
    batch_size: int,
    max_data: int = 100,
) -> Optional[Dict]:
    dynamic_axis = {"inputs": [{}] * len(input_sizes), "outputs": []}
    output_sizes = []
    for i, (input_tensors, y) in enumerate(dataset):
        if i >= max_data:
            break
        inspect_dynamic_size(
            input_tensors, input_sizes, batch_size, dynamic_axis["inputs"]
        )
        outputs = tuple(run_tf_model(tf_model, input_tensors))
        if i == 0:
            dynamic_axis["outputs"] = [{}] * len(outputs)
            output_sizes = [tuple(output.shape[1:]) for output in outputs]
        inspect_dynamic_size(
            outputs, output_sizes, batch_size, dynamic_axis["outputs"]
        )
    if any(
        len(x) > 0 for x in (dynamic_axis["inputs"] + dynamic_axis["outputs"])
    ):
        return dynamic_axis
    return None


def _extract_info_from_data(
    tf_model: tf.Module,
    dataset: List[Tuple[Tuple[tf.Tensor, ...], Any]],
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    input_types: List[str],
    dynamic_axis: Dict,
):
    input_row, _ = dataset[0]
    batch_size = ifnone(batch_size, int(input_row[0].shape[0]))
    input_sizes = ifnone(input_sizes, [tuple(x.shape[1:]) for x in input_row])
    input_types = ifnone(
        input_types, ["int" if x.dtype == int else "float" for x in input_row]
    )
    dynamic_axis = ifnone(
        dynamic_axis,
        _extract_dynamic_axis(tf_model, dataset, input_sizes, batch_size),
    )
    return batch_size, input_sizes, input_types, dynamic_axis


def optimize_tf_model(
    model: Union[tf.Module, tf.keras.Model],
    save_dir: str,
    dataset: List[Tuple[Tuple[tf.Tensor, ...], Any]] = None,
    batch_size: int = None,
    input_sizes: List[Tuple[int, ...]] = None,
    input_types: List[str] = None,
    extra_input_info: List[Dict] = None,
    dynamic_axis: Dict = None,
    perf_loss_ths: float = None,
    perf_metric: Union[str, Callable] = None,
    ignore_compilers: List[str] = None,
    custom_optimizers: List[BaseOptimizer] = None,
):
    """Basic function for optimizing a tensorflow model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.

    Args:
        model (tf.Module or keras.Model): Model that needs optimization.
        save_dir (str): Path to the directory where saving the final model.
        dataset (List, optional):  Dataset containing data in the form of
            (xs, y) where xs are tuples of Tensors and ys can be whatever
            needed for computing the selected metric at quantization time.
            The data will be used for extracting all the data related
            information or completing the missing information in the case of
            partially given ones. If no data is given, both the `batch_size`
            and the `input_sizes` must be passed by the user.
        batch_size (int, optional): The model batch size.
        input_sizes (List[Tuple]], optional): List containing the size of all
            the input tensors of the model. Note that even just a single
            tensor is needed as model input, this field must be a list
            containing (in the exposed case) a single element). The tuple must
            contain all then input tensor dimensions excluding the batch size.
            This means that the final input tensor size will be considered as
            `(batch_size, *input_tensor_size)`, where `input_tensor_size` is
            one list element of `input_sizes`.
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
        perf_loss_ths (float, optional): Tolerated relative error for
            performing approximation techniques before compiling the model.
            If no value is given, no optimization will be performed. Note that
            it will not be used for compilers using the torch API when
            `use_torch_api` is `True`. Just dynamic quantization will be
            performed, since no data is given as input.
        perf_metric (Union[Callable, str], optional): The metric to
            be used for accepting or refusing a precision-reduction
            optimization proposal. If none is given but a `perf_loss_ths` is
            received, the `nebullvm.measure.compute_relative_difference`
            metric will be used as default one. A user-defined metric can
            be passed as function accepting as inputs two tuples of tensors
            (produced by the baseline and the quantized model) and the related
            original labels.
            For more information see
            `nebullvm.measure.compute_relative_difference` and
            `nebullvm.measure.compute_accuracy_drop`. `perf_metric`
            accepts as value also a string containing the metric name. At the
            current stage the supported metrics are `"precision"` and
            `"accuracy"`.
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
    if dataset is not None:
        (
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        ) = _extract_info_from_data(
            model, dataset, batch_size, input_sizes, input_types, dynamic_axis
        )
        input_data = DataManager(dataset)
    else:
        input_data = None
    if isinstance(perf_metric, str):
        perf_metric = QUANTIZATION_METRIC_MAP.get(perf_metric)
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
    input_tfms = MultiStageTransformation([])
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
        model_optimized = model_optimizer.optimize(
            onnx_model=str(onnx_path),
            output_library=dl_library,
            model_params=model_params,
            input_tfms=input_tfms,
            perf_loss_ths=perf_loss_ths,
            perf_metric=perf_metric,
            input_data=input_data,
        )
        model_optimized.save(save_dir)
    return model_optimized.load(save_dir)
