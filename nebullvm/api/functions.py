import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence, Union, Dict, Callable

import torch.nn
import tensorflow as tf

from nebullvm.api.frontend.onnx import extract_info_from_np_data
from nebullvm.api.frontend.tf import extract_info_from_tf_data
from nebullvm.api.frontend.torch import extract_info_from_torch_data
from nebullvm.api.utils import QUANTIZATION_METRIC_MAP
from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.converters.converters import CrossConverter
from nebullvm.pipelines.steps import build_pipeline_from_model
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.utils.onnx import get_output_sizes_onnx
from nebullvm.utils.tf import get_outputs_sizes_tf
from nebullvm.utils.torch import get_outputs_sizes_torch


logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_dl_framework(model: Any):
    if isinstance(model, torch.nn.Module):
        return DeepLearningFramework.PYTORCH
    elif isinstance(model, tf.Module):
        return DeepLearningFramework.TENSORFLOW
    elif isinstance(model, str):
        return DeepLearningFramework.NUMPY
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


def _extract_info_from_data(
    model: Any,
    input_data: DataManager,
    dl_framework: DeepLearningFramework,
    dynamic_info: Dict,
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
    )
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=[
            {"size": size, "dtype": dtype}
            for size, dtype in zip(input_sizes, input_types)
        ],
        output_sizes=OUTPUT_SIZE_COMPUTATION_DICT[dl_framework](
            model, input_data[0][0]
        ),
        dynamic_info=dynamic_info,
    )
    return model_params


def optimize_model(
    model: Any,
    input_data: Union[Iterable, Sequence],
    metric_drop_ths: float,
    metric: Union[str, Callable],
    optimization_time: str,
    dynamic_info: Dict,
    config_file: str,
    **kwargs,
):
    dl_framework = _get_dl_framework(model)
    FEEDBACK_COLLECTOR.start_collection(model, framework=dl_framework)
    if isinstance(metric, str):
        metric = QUANTIZATION_METRIC_MAP.get(metric)
    needs_conversion_to_hf = False
    if isinstance(model, tuple):  # Huggingface model
        from nebullvm.api.huggingface import convert_hf_model

        (
            model,
            input_data,
            input_names,
            output_structure,
            output_type,
        ) = convert_hf_model(model, input_data, **kwargs)
        needs_conversion_to_hf = True
    if _check_input_data(input_data):
        input_data = DataManager(input_data)
    else:
        input_data = DataManager.from_iterable(input_data)
    model_params = _extract_info_from_data(
        model,
        input_data,
        dl_framework,
        dynamic_info,
    )
    converter = CrossConverter()
    optimized_models = []
    with TemporaryDirectory as tmp_dir:
        tmp_dir = Path(tmp_dir)
        models = converter.convert(model, model_params, tmp_dir, input_data)
        ignore_compilers = []
        for model in models:
            input_tfms = MultiStageTransformation([])
            pipeline = build_pipeline_from_model(
                model, optimization_time, metric_drop_ths, metric, config_file
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
            )
            ignore_compilers = output_dict["ignore_compilers"]
            optimized_models.extend(output_dict["optimized_models"])

    optimized_models.sort(key=lambda x: x[1], reverse=False)
    optimal_model = optimized_models[0][0]
    FEEDBACK_COLLECTOR.send_feedback()
    if optimal_model is None:
        logger.warning(
            "No optimized model has been created. This is likely due to a "
            "bug in Nebullvm. Please open an issue and report in details "
            "your use case."
        )
        return optimal_model
    if needs_conversion_to_hf:
        from nebullvm.api.huggingface import HuggingFaceInferenceLearner

        optimal_model = HuggingFaceInferenceLearner(
            core_inference_learner=optimal_model,
            output_structure=output_structure,
            input_names=input_names,
            output_type=output_type,
        )
    return optimal_model
