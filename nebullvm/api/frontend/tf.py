import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union, Dict, Optional, Callable, Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from nebullvm.api.utils import (
    ifnone,
    inspect_dynamic_size,
    QUANTIZATION_METRIC_MAP,
)
from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    InputInfo,
    ModelCompiler,
    QuantizationType,
)
from nebullvm.converters import ONNXConverter
from nebullvm.inference_learners import TensorflowBaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.tensorflow import TensorflowBackendOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.utils.tf import (
    get_outputs_sizes_tf,
    create_model_inputs_tf,
    run_tf_model,
)
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def extract_info_from_tf_data(
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
            optimization proposal. If none is given but a `metric_drop_ths` is
            received, the `nebullvm.measure.compute_relative_difference`
            metric will be used as default one. A user-defined metric can
            be passed as function accepting as inputs two tuples of tensors
            (produced by the baseline and the quantized model) and the related
            original labels.
            For more information see
            `nebullvm.measure.compute_relative_difference` and
            `nebullvm.measure.compute_accuracy_drop`. `metric`
            accepts as value also a string containing the metric name. At the
            current stage the supported metrics are `"numeric_precision"` and
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
            and it gives as output `tf.Tensor` s.
    """
    warnings.warn(
        "Deprecated: The usage of the tensorflow api is deprecated. "
        "`optimize_tf_model`will be removed from the next release. "
        "Use `optimize_model` instead."
    )
    if dataset is not None:
        (
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        ) = extract_info_from_tf_data(
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
    FEEDBACK_COLLECTOR.start_collection(model, framework=dl_library)
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
        logger=logger,
    )
    with TemporaryDirectory() as tmp_dir:
        logger.info("Running Optimization using tensorflow interface (1/3)")
        if perf_loss_ths is not None:
            q_types = [
                None,
                QuantizationType.DYNAMIC,
                QuantizationType.HALF,
            ]
            if dataset is not None:
                q_types.append(QuantizationType.STATIC)
        else:
            q_types = [None]
        torch_res = [
            _tf_api_optimization(
                model,
                model_params,
                perf_loss_ths,
                q_type,
                input_tfms,
                False,
                input_data,
            )
            for q_type in tqdm(q_types)
        ]
        (tf_api_model, tf_api_latency, used_compilers,) = sorted(
            torch_res, key=lambda x: x[1]
        )[0]
        ignore_compilers.extend(used_compilers)
        logger.info("Running Optimization using ONNX interface (2/3)")
        onnx_path = model_converter.convert(
            model, model_params.input_sizes, Path(tmp_dir)
        )
        model_optimized = model_optimizer.optimize(
            model=str(onnx_path),
            output_library=dl_library,
            model_params=model_params,
            input_tfms=input_tfms,
            metric_drop_ths=perf_loss_ths,
            metric=perf_metric,
            input_data=input_data,
        )
        logger.info("Running comparison between optimized models (3/3).")
        model_optimized = _compare_optimized_models(
            model_optimized, tf_api_model, tf_api_latency
        )
        if model_optimized is None:
            raise RuntimeError(
                "No valid compiled model has been produced. "
                "Look at the logs for further information about the failure."
            )
        model_optimized.save(save_dir)
    FEEDBACK_COLLECTOR.send_feedback()
    return model_optimized.load(save_dir)


def _compare_optimized_models(
    new_model: TensorflowBaseInferenceLearner,
    previous_best_model: TensorflowBaseInferenceLearner,
    previous_latency: float,
) -> TensorflowBaseInferenceLearner:
    if new_model is not None:
        new_latency = compute_optimized_running_time(new_model)
        if new_latency < previous_latency:
            return new_model
    return previous_best_model


def _get_optimizers_supporting_tf_api(use_extra_compilers: bool):
    if use_extra_compilers:
        logger.warning(
            "No compiler found supporting the tensorflow interface."
        )
    return [(ModelCompiler.TFLITE, TensorflowBackendOptimizer(logger=logger))]


def _tf_api_optimization(
    model: tf.Module,
    model_params: ModelParams,
    quantization_ths: float,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    use_extra_compilers: bool,
    input_data: DataManager,
) -> Tuple[Optional[TensorflowBaseInferenceLearner], float, List]:
    used_compilers = []
    best_tf_opt_model = None
    best_latency = np.inf
    for compiler, optimizer in tqdm(
        _get_optimizers_supporting_tf_api(use_extra_compilers)
    ):
        try:
            if hasattr(optimizer, "optimize_from_tf"):
                candidate_model = optimizer.optimize_from_tf(
                    torch_model=model,
                    model_params=model_params,
                    perf_loss_ths=quantization_ths
                    if quantization_type is not None
                    else None,
                    quantization_type=quantization_type,
                    input_tfms=input_tfms.copy(),
                    input_data=input_data,
                )
            else:
                candidate_model = optimizer.optimize(
                    model=model,
                    output_library=DeepLearningFramework.PYTORCH,
                    model_params=model_params,
                    metric_drop_ths=quantization_ths
                    if quantization_type is not None
                    else None,
                    quantization_type=quantization_type,
                    input_tfms=input_tfms.copy(),
                    input_data=input_data,
                )
            candidate_latency = compute_optimized_running_time(candidate_model)
            if candidate_latency < best_latency:
                best_latency = candidate_latency
                best_tf_opt_model = candidate_model
            FEEDBACK_COLLECTOR.store_compiler_result(
                compiler=compiler,
                q_type=quantization_type,
                metric_drop_ths=quantization_ths,
                latency=candidate_latency,
                pipeline_name="tensorflow",
            )
            used_compilers.append(compiler)
        except Exception as ex:
            warnings.warn(
                f"Compilation failed with torch interface of {compiler}. "
                f"Got error {ex}. If possible the compilation will be "
                f"re-scheduled with the ONNX interface. Please consult the "
                f"documentation for further info or open an issue on GitHub "
                f"for receiving assistance."
            )
            FEEDBACK_COLLECTOR.store_compiler_result(
                compiler=compiler,
                q_type=quantization_type,
                metric_drop_ths=quantization_ths,
                latency=None,
                pipeline_name="tensorflow",
            )
    return best_tf_opt_model, best_latency, used_compilers
