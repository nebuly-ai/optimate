import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Dict, Optional, Callable, Union, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nebullvm.api.utils import (
    check_inputs,
    ifnone,
    inspect_dynamic_size,
    QUANTIZATION_METRIC_MAP,
)
from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    ModelCompiler,
    InputInfo,
    QuantizationType,
)
from nebullvm.converters import ONNXConverter
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR
from nebullvm.utils.torch import (
    get_outputs_sizes_torch,
    create_model_inputs_torch,
    run_torch_model,
)
from nebullvm.inference_learners.base import PytorchBaseInferenceLearner
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    ApacheTVMOptimizer,
    BaseOptimizer,
)
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _extract_dynamic_axis(
    torch_model: torch.nn.Module,
    dataloader: DataLoader,
    input_sizes: List[Tuple[int, ...]],
    batch_size: int,
    max_data: int = 100,
) -> Optional[Dict]:
    dynamic_axis = {"inputs": [{}] * len(input_sizes), "outputs": []}
    output_sizes = []
    for i, (input_tensors, y) in enumerate(dataloader):
        if i >= max_data:
            break
        inspect_dynamic_size(
            input_tensors, input_sizes, batch_size, dynamic_axis["inputs"]
        )
        outputs = tuple(run_torch_model(torch_model, input_tensors))
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


def extract_info_from_torch_data(
    model: torch.nn.Module,
    dataloader: Union[DataLoader, Sequence],
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    input_types: List[str],
    dynamic_axis: Dict,
):
    input_row, _ = (
        dataloader[0]
        if isinstance(dataloader, Sequence)
        else next(iter(dataloader))
    )
    batch_size = ifnone(batch_size, int(input_row[0].shape[0]))
    input_sizes = ifnone(input_sizes, [tuple(x.shape[1:]) for x in input_row])
    input_types = ifnone(
        input_types,
        [
            "int" if isinstance(x.cpu(), torch.LongTensor) else "float"
            for x in input_row
        ],
    )
    dynamic_axis = ifnone(
        dynamic_axis,
        _extract_dynamic_axis(model, dataloader, input_sizes, batch_size),
    )
    return batch_size, input_sizes, input_types, dynamic_axis


def optimize_torch_model(
    model: torch.nn.Module,
    save_dir: str,
    dataloader: Union[DataLoader, Sequence] = None,
    batch_size: int = None,
    input_sizes: List[Tuple[int, ...]] = None,
    input_types: List[str] = None,
    extra_input_info: List[Dict] = None,
    use_torch_api: bool = False,
    dynamic_axis: Dict = None,
    perf_loss_ths: float = None,
    perf_metric: Union[str, Callable] = None,
    ignore_compilers: List[str] = None,
    custom_optimizers: List[BaseOptimizer] = None,
) -> PytorchBaseInferenceLearner:
    """Basic function for optimizing a torch model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.

    Args:
        model (torch.nn.Module): Pytorch model that needs optimization.
        save_dir (str): Path to the directory where saving the final model.
        dataloader (DataLoader, optional): Data loader to be used for loading
            the user uploaded data. The data will be used for extracting all
            the data related information or completing the missing information
            in the case of partially given ones. If not given, both the
            `batch_size` and the `input_sizes` must be given by the user.
        batch_size (int, optional): The model batch size. Note that nebullvm
            does not support at the moment dynamic batch size, so a valid
            input should be given.
        input_sizes (List[Tuple]], optional): List containing the size of all
            the input tensors of the model. Note that even just a single
            tensor is needed as model input, this field must be a list
            containing (in the exposed case) a single element).
            The tuple must contain all the input tensor dimensions excluding
            the batch size. This means that the final input tensor size will
            be considered as (batch_size, *input_tensor_size)`, where
            `input_tensor_size` is one list element of `input_sizes`.
        input_types (List[str], optional): List of input types. If no value is
            given all the inputs will be considered as float type. The
            supported string values are "int" and "float".
        extra_input_info (List[Dict], optional): List of extra information
            needed for defining the input tensors, e.g. max_value and min_value
            the tensors can get.
        use_torch_api (bool): Parameter for using the torch api of compilers
            when available. The actual implementation supports only the torch
            interface for TVM. Note that when running the torch interface
            nebullvm will ignore the ONNX one once the torch implementation
            succeeds. Clearly, in case of failure of the torch API, a second
            tentative will be done with the ONNX interface.
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
        ignore_compilers (List[str], optional): List of DL compilers we want
            to ignore while running the optimization. Compiler name should be
            one between "tvm", "tensor RT", "openvino" and "onnxruntime".
        custom_optimizers (List[BaseOptimizer], optional): List of optimizers
            which can be used for producing InferenceLearners, i.e. models
            optimized for inference. This list is useful when some compilers,
            not included in the base version of nebullvm or not used
            by default, which are specific for a particular tasks, need
            to be used.

    Returns:
        PytorchBaseInferenceLearner: Optimized model usable with the classical
            Pytorch interface. Note that as a torch model it takes as input
            and it gives as output `torch.Tensor` s.
    """
    warnings.warn(
        "Deprecated: The usage of the torch api is deprecated. "
        "`optimize_torch_model`will be removed from the next release. "
        "Use `optimize_model` instead."
    )
    check_inputs(
        input_data=dataloader, batch_size=batch_size, input_sizes=input_sizes
    )
    if isinstance(perf_metric, str):
        perf_metric = QUANTIZATION_METRIC_MAP.get(perf_metric)
    if dataloader is not None:
        (
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        ) = extract_info_from_torch_data(
            model,
            dataloader,
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        )
        input_data = (
            DataManager.from_iterable(dataloader)
            if isinstance(dataloader, DataLoader)
            else DataManager(dataloader)
        )
    else:
        input_data = None
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
    dl_library = DeepLearningFramework.PYTORCH

    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=input_infos,
        output_sizes=get_outputs_sizes_torch(
            model,
            input_tensors=list(input_data.get_list(1)[0])
            if input_data is not None
            else create_model_inputs_torch(batch_size, input_infos),
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
    with TemporaryDirectory() as tmp_dir:
        logger.info("Running Optimization using torch interface (1/3)")
        if perf_loss_ths is not None:
            q_types = [
                None,
                QuantizationType.DYNAMIC,
                QuantizationType.HALF,
            ]
            if dataloader is not None:
                q_types.append(QuantizationType.STATIC)
        else:
            q_types = [None]
        torch_res = [
            _torch_api_optimization(
                model,
                model_params,
                perf_loss_ths,
                q_type,
                input_tfms,
                use_torch_api,
                input_data,
            )
            for q_type in tqdm(q_types)
        ]
        (torch_api_model, torch_api_latency, used_compilers,) = sorted(
            torch_res, key=lambda x: x[1]
        )[0]
        ignore_compilers.extend(used_compilers)
        logger.info("Running Optimization using ONNX interface (2/3)")
        model_converter = ONNXConverter()
        model_optimizer = MultiCompilerOptimizer(
            ignore_compilers=ignore_compilers,
            extra_optimizers=custom_optimizers,
            debug_mode=int(os.environ.get("DEBUG_MODE", "0")) > 0,
            logger=logger,
        )
        if model_optimizer.usable:
            onnx_path = model_converter.convert(
                model, model_params, Path(tmp_dir), input_data
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
        else:
            model_optimized = None
        logger.info("Running comparison between optimized models (3/3).")
        model_optimized = _compare_optimized_models(
            model_optimized,
            torch_api_model,
            torch_api_latency,
        )
        if model_optimized is None:
            raise RuntimeError(
                "No valid compiled model has been produced. "
                "Look at the logs for further information about the failure."
            )
        model_optimized.save(save_dir)
    FEEDBACK_COLLECTOR.send_feedback()
    return model_optimized.load(save_dir)


def _get_optimizers_supporting_torch_api(
    use_extra_compilers: bool,
) -> List[Tuple[ModelCompiler, BaseOptimizer]]:
    optimizers = [
        (ModelCompiler.TORCHSCRIPT, PytorchBackendOptimizer(logger=logger)),
    ]
    if use_extra_compilers:
        optimizers.append(
            (ModelCompiler.APACHE_TVM, ApacheTVMOptimizer(logger=logger))
        )
    return optimizers


def _torch_api_optimization(
    model: torch.nn.Module,
    model_params: ModelParams,
    quantization_ths: float,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    use_extra_compilers: bool,
    input_data: DataManager,
) -> Tuple[Optional[PytorchBaseInferenceLearner], float, List]:
    used_compilers = []
    best_torch_opt_model = None
    best_latency = np.inf
    for compiler, optimizer in tqdm(
        _get_optimizers_supporting_torch_api(use_extra_compilers)
    ):
        try:
            if hasattr(optimizer, "optimize_from_torch"):
                candidate_model = optimizer.optimize_from_torch(
                    torch_model=model,
                    model_params=model_params,
                    metric_drop_ths=quantization_ths
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
                best_torch_opt_model = candidate_model
            FEEDBACK_COLLECTOR.store_compiler_result(
                compiler=compiler,
                q_type=quantization_type,
                metric_drop_ths=quantization_ths,
                latency=candidate_latency,
                pipeline_name="pytorch",
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
                pipeline_name="pytorch",
            )
    return best_torch_opt_model, best_latency, used_compilers


def _compare_optimized_models(
    new_model: PytorchBaseInferenceLearner,
    previous_best_model: PytorchBaseInferenceLearner,
    previous_latency: float,
) -> PytorchBaseInferenceLearner:
    if new_model is not None:
        new_latency = compute_optimized_running_time(new_model)
        if new_latency < previous_latency:
            return new_model
    return previous_best_model
