import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Iterable,
    Sequence,
    Union,
    Dict,
    Callable,
    List,
    Optional,
)

from nebullvm.api.frontend.onnx import extract_info_from_np_data
from nebullvm.api.frontend.tf import extract_info_from_tf_data
from nebullvm.api.frontend.torch import extract_info_from_torch_data
from nebullvm.api.huggingface import convert_hf_model, is_dict_type
from nebullvm.api.utils import QUANTIZATION_METRIC_MAP
from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    ModelCompiler,
    ModelCompressor,
    OptimizationTime,
)
from nebullvm.config import QUANTIZATION_DATA_NUM, TRAIN_TEST_SPLIT_RATIO
from nebullvm.converters.converters import CrossConverter
from nebullvm.measure import (
    compute_torch_latency,
    compute_tf_latency,
    compute_onnx_latency,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import Module
from nebullvm.pipelines.steps import build_pipeline_from_model
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.utils.general import gpu_is_available
from nebullvm.utils.onnx import (
    get_output_sizes_onnx,
    run_onnx_model,
)
from nebullvm.utils.tf import (
    get_outputs_sizes_tf,
    run_tf_model,
)
from nebullvm.utils.torch import (
    get_outputs_sizes_torch,
    run_torch_model,
)

logger = logging.getLogger("nebullvm_logger")


def _get_dl_framework(model: Any):
    if isinstance(model, Module):
        return DeepLearningFramework.PYTORCH
    elif isinstance(model, tf.Module) and model is not None:
        return DeepLearningFramework.TENSORFLOW
    elif isinstance(model, str):
        if Path(model).is_file():
            return DeepLearningFramework.NUMPY
        else:
            raise FileNotFoundError(
                f"No file '{model}' found, please provide a valid path to "
                f"a model."
            )
    else:
        raise TypeError(f"Model type {type(model)} not supported.")


def _check_input_data(input_data: Union[Iterable, Sequence]):
    try:
        input_data[0]
    except:  # noqa E722
        return False
    else:
        return True


INFO_EXTRACTION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: extract_info_from_torch_data,
    DeepLearningFramework.TENSORFLOW: extract_info_from_tf_data,
    DeepLearningFramework.NUMPY: extract_info_from_np_data,
}

OUTPUT_SIZE_COMPUTATION_DICT: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: get_outputs_sizes_torch,
    DeepLearningFramework.TENSORFLOW: get_outputs_sizes_tf,
    DeepLearningFramework.NUMPY: get_output_sizes_onnx,
}

COMPUTE_OUTPUT_FRAMEWORK: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: run_torch_model,
    DeepLearningFramework.TENSORFLOW: run_tf_model,
    DeepLearningFramework.NUMPY: run_onnx_model,
}

COMPUTE_LATENCY_FRAMEWORK: Dict[DeepLearningFramework, Callable] = {
    DeepLearningFramework.PYTORCH: compute_torch_latency,
    DeepLearningFramework.TENSORFLOW: compute_tf_latency,
    DeepLearningFramework.NUMPY: compute_onnx_latency,
}


def _extract_info_from_data(
    model: Any,
    input_data: DataManager,
    dl_framework: DeepLearningFramework,
    dynamic_info: Optional[Dict],
    device: str,
):
    batch_size, input_sizes, input_types, dynamic_info = INFO_EXTRACTION_DICT[
        dl_framework
    ](
        model,
        input_data,
        batch_size=None,
        input_sizes=None,
        input_types=None,
        dynamic_axis=dynamic_info,
        device=device,
    )
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=[
            {"size": size, "dtype": dtype}
            for size, dtype in zip(input_sizes, input_types)
        ],
        output_sizes=OUTPUT_SIZE_COMPUTATION_DICT[dl_framework](
            model, input_data[0][0], device
        ),
        dynamic_info=dynamic_info,
    )
    return model_params


def _is_huggingface_data(data_sample: Any) -> bool:
    if is_dict_type(data_sample):
        return True
    elif isinstance(data_sample, str):
        return True
    elif isinstance(data_sample[0], str):
        return True
    return False


def _benchmark_original_model(
    model: Any,
    input_data: DataManager,
    dl_framework: DeepLearningFramework,
    device: str,
    compute_output: bool = False,
):
    outputs = None

    logger.info("Benchmark performance of original model")

    if compute_output:
        outputs = [
            tuple(
                COMPUTE_OUTPUT_FRAMEWORK[dl_framework](
                    model, list(input_tensors[0]), device
                )
            )
            for input_tensors in input_data
        ]

    inputs = input_data.get_list(QUANTIZATION_DATA_NUM)

    device = "cuda" if device == "gpu" else "cpu"
    latency, _ = COMPUTE_LATENCY_FRAMEWORK[dl_framework](inputs, model, device)
    logger.info(f"Original model latency: {latency} sec/iter")

    return outputs, latency


def _map_compilers_and_compressors(ignore_list: List, enum_class: Callable):
    if ignore_list is None:
        ignore_list = []
    else:
        ignore_list = [enum_class(element) for element in ignore_list]
    return ignore_list


def _check_device(
    device: Optional[str], dl_framework: DeepLearningFramework
) -> str:
    if device is None:
        if gpu_is_available(dl_framework):
            device = "gpu"
        else:
            device = "cpu"
    else:
        if device.upper() == "gpu":
            if not gpu_is_available(dl_framework):
                logger.warning(
                    "Selected GPU device but no available GPU found on this "
                    "platform. CPU will be used instead. Please make sure "
                    "that the gpu is installed and can be used by your "
                    "framework."
                )
                device = "cpu"
            else:
                device = "gpu"
        else:
            device = "cpu"

    logger.info(f"Running Nebullvm optimization on {device.upper()}")

    return device


def optimize_model(
    model: Any,
    input_data: Union[Iterable, Sequence],
    metric_drop_ths: float = None,
    metric: Union[str, Callable] = None,
    optimization_time: str = "constrained",
    dynamic_info: Dict = None,
    config_file: str = None,
    ignore_compilers: List[str] = None,
    ignore_compressors: List[str] = None,
    store_latencies: bool = False,
    device: Optional[str] = None,
    **kwargs,
):
    """Optimize the input model regardless of the framework it was used for
    implementing it. The optimized model given as output will share with the
    input one the same API, i.e. the optimized model will have the same
    interface as the original one.

    Args:
        model (Any): The input model.
        input_data (Iterable or Sequence): Input data to be used for
            optimizing the model. Note that if 'unconstrained' is selected as
            `optimization_time`, it would be beneficial to provide at least 100
            data samples in order to use all the techniques supported by
            Nebullvm. The data can be given in either as sequence (data can be
            accessed by "element", e.g. `data[i]`) or iterable (data needs to
            be accessed with loop, e.g. `for x in data`). PyTorch, TensorFlow
            and Onnx respectively accept input tensor in `torch.Tensor`,
            `tf.Tensor` and `np.ndarray` formats. Note that the each input
            sample must be a tuple containing a tuple as first element, the
            `inputs`, and the `label` as second element. The `inputs` needs to
            be passed as tuple even if a single input is needed by the model
            (in this case the `inputs` tuple will contain just an element).
            HuggingFace models can take as data samples both dictionaries or
            strings. Strings will then be converted in data samples using the
            HuggingFace tokenizer which must be given as input when just a
            list of string is provided as input_data (tokenizers can be passed
            as extra arguments of this function using the keyword `tokenizer`).
        metric_drop_ths (float, optional): Maximum reduction in the
            selected metric accepted. No model with an higher error will be
            accepted, i.e. all optimized model having a larger error respect to
            the original one will be discarded, without even considering their
            possible speed-up. Default: None, i.e. no drop in metric accepted.
        metric (Union[Callable, str], optional): The metric to
            be used for accepting or refusing a precision-reduction
            optimization proposal. If none is given but a `metric_drop_ths` is
            received, the `nebullvm.measure.compute_relative_difference`
            metric will be used as default one. A user-defined metric can
            be passed as function accepting as inputs two tuples of tensors
            (produced by the baseline and the optimized model) and the related
            original labels.
            For more information see
            `nebullvm.measure.compute_relative_difference` and
            `nebullvm.measure.compute_accuracy_drop`. `metric`
            accepts as value also a string containing the metric name. At the
            current stage the supported metrics are `"numeric_precision"` and
            `"accuracy"`. Default: `"numeric_precision"`
        optimization_time (OptimizationTime, optional): The optimization time
            mode. It can be either 'constrained' or 'unconstrained'. For
            'constrained' mode just compilers and precision reduction
            techniques are used (no compression). 'Unconstrained' optimization
            allows the usage of more time consuming techniques as pruning and
            distillation. Note that for using many of the sophisticated
            techniques in the 'unconstrained' optimization, a small fine-tuning
            of the model will be needed. Thus we highly recommend to give as
            input_data at least 100 samples for when selecting 'unconstrained'
            optimization. Default: 'constrained'.
        dynamic_info (Dict, optional): Dictionary containing info about the
            dynamic axis. It should contain as keys both "inputs" and "outputs"
            and as values two lists of dictionaries where each dictionary
            represents the dynamic axis information for an input/output tensor.
            The inner dictionary should have as key an integer, i.e. the
            dynamic axis (considering also the batch size) and as value a
            string giving a "tag" to it, e.g. "batch_size". Default: None
        config_file (str, optional): Configuration file containing the
            parameters needed for defining the CompressionStep in the pipeline.
            Default: None.
        ignore_compilers (List, optional): List containing the compilers to be
            ignored during the OptimizerStep. Default: None.
        ignore_compressors (List, optional): List containing the compressors
            to be ignored during the CompressionStep. Default: None.
        store_latencies (bool, optional): Parameter that allows to save the
            latency for each compiler used by nebullvm. Default: False.
        device (str, optional): Device used, can be 'cpu' or 'gpu'. If not
            set, gpu will be used if available, otherwise cpu. Default: None

    Returns:
        InferenceLearner: Optimized version of the input model having the same
            interface, imported by its original framework. For instance a
            Pytorch model, when optimized, will return an InferenceLearner
            object that can be call exactly as a PyTorch model (either
            with `model.forward(input)` and `model(input)`), i.e. it will
            take as input and it will return `torch.Tensor`s.
    """
    dl_framework = _get_dl_framework(model)
    device = _check_device(device, dl_framework)
    optimization_time = OptimizationTime(optimization_time)
    FEEDBACK_COLLECTOR.start_collection(
        model, framework=dl_framework, device=device
    )
    if metric_drop_ths is not None and metric_drop_ths <= 0:
        metric_drop_ths = None
    elif metric_drop_ths is not None and metric is None:
        metric = "numeric_precision"
    if isinstance(metric, str):
        metric = QUANTIZATION_METRIC_MAP.get(metric)
    needs_conversion_to_hf = False
    if _is_huggingface_data(input_data[0]):
        (
            model,
            input_data,
            input_names,
            output_structure,
            output_type,
        ) = convert_hf_model(model, input_data, device, **kwargs)
        needs_conversion_to_hf = True
    if _check_input_data(input_data):
        input_data = DataManager(input_data)
    else:
        input_data = DataManager.from_iterable(input_data)
    input_data.split(TRAIN_TEST_SPLIT_RATIO)
    model_params = _extract_info_from_data(
        model,
        input_data,
        dl_framework,
        dynamic_info,
        device,
    )
    converter = CrossConverter()
    optimized_models = []
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir) / "fp32"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        models = converter.convert(
            model, model_params, tmp_dir, device, input_data
        )

        ignore_compilers = _map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        ignore_compressors = _map_compilers_and_compressors(
            ignore_compressors, ModelCompressor
        )

        # Benchmark original model
        model_outputs, orig_latency = _benchmark_original_model(
            model=model,
            input_data=input_data.get_split("test"),
            dl_framework=dl_framework,
            device=device,
            compute_output=True,
        )

        # Store original model result
        FEEDBACK_COLLECTOR.store_compiler_result(
            compiler=dl_framework,
            q_type=None,
            metric_drop_ths=metric_drop_ths,
            latency=orig_latency,
            pipeline_name="original",
        )

        for model in models:
            input_tfms = MultiStageTransformation([])
            pipeline = build_pipeline_from_model(
                model,
                optimization_time,
                metric_drop_ths,
                config_file,
            )
            output_dict = pipeline.run(
                model=model,
                input_data=input_data,
                metric_drop_ths=metric_drop_ths,
                metric=metric,
                output_library=dl_framework,
                model_params=model_params,
                input_tfms=input_tfms,
                ignore_compilers=ignore_compilers,
                ignore_compressors=ignore_compressors,
                optimization_time=optimization_time,
                model_outputs=model_outputs,
                device=device,
            )
            ignore_compilers = output_dict["ignore_compilers"]
            optimized_models.extend(output_dict["optimized_models"])

    optimized_models.sort(key=lambda x: x[1], reverse=False)
    FEEDBACK_COLLECTOR.send_feedback(store_latencies)

    if len(optimized_models) < 1 or optimized_models[0][0] is None:
        logger.warning(
            "No optimized model has been created. This is likely due to a "
            "bug in Nebullvm. Please open an issue and report in details "
            "your use case."
        )
        return None

    optimal_model = optimized_models[0][0]

    logger.info("--- Nebullvm results ---")
    logger.info(f"Original model latency: {orig_latency} sec/iter")
    logger.info(f"Optimized model latency: {optimized_models[0][1]} sec/iter")
    logger.info(
        "Estimated speedup: {:.2f}x".format(
            orig_latency / optimized_models[0][1]
        )
    )

    if needs_conversion_to_hf:
        from nebullvm.api.huggingface import HuggingFaceInferenceLearner

        optimal_model = HuggingFaceInferenceLearner(
            core_inference_learner=optimal_model,
            output_structure=output_structure,
            input_names=input_names,
            output_type=output_type,
        )
    return optimal_model
