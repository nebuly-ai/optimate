import logging
from typing import (
    Union,
    Iterable,
    Sequence,
    Callable,
    Dict,
    List,
    Optional,
)

from nebullvm.config import DEFAULT_METRIC_DROP_THS
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.logger import debug_mode_enabled, LoggingContext

from speedster.root_op import SpeedsterRootOp

from nebullvm.tools.utils import check_device


def optimize_model(
    model: Union[torch.nn.Module, tf.Module, str],
    input_data: Union[Iterable, Sequence],
    metric_drop_ths: float = DEFAULT_METRIC_DROP_THS,
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
        model (Union[torch.Module, tf.Module, str]): The input model. It can be
            a torch or tensorflow model or a path to an onnx saved model.
        input_data (Iterable or Sequence): Input data to be used for
            optimizing the model. Note that if 'unconstrained' is selected as
            `optimization_time`, it would be beneficial to provide at least 100
            data samples in order to use all the techniques supported by
            Nebullvm. The data can be given in either as sequence (data can be
            accessed by "element", e.g. `data[i]`) or iterable (data needs to
            be accessed with loop, e.g. `for x in data`). PyTorch, TensorFlow
            and Onnx respectively accept input tensor in `torch.Tensor`,
            `tf.Tensor` and `np.ndarray` formats. Note that each input
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
            selected metric accepted. No model with a higher error will be
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
            allows the usage of more time-consuming techniques as pruning and
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
            ignored during the OptimizerStep. The compiler name should be one
            among tvm, tensor RT, openvino, onnxruntime, deepsparse, tflite,
            bladedisc, torchscript, intel_neural_compressor. Default: None.
        ignore_compressors (List, optional): List containing the compressors
            to be ignored during the CompressionStep. The compiler name should
            be one among . Default: None.
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
    root_op = SpeedsterRootOp()
    device = check_device(device)

    disable_log = True if not debug_mode_enabled() else False

    with LoggingContext(logging.getLogger(), disabled=disable_log):
        root_op.to(device).execute(
            model=model,
            input_data=input_data,
            metric_drop_ths=metric_drop_ths,
            metric=metric,
            optimization_time=optimization_time,
            dynamic_info=dynamic_info,
            config_file=config_file,
            ignore_compilers=ignore_compilers,
            ignore_compressors=ignore_compressors,
            store_latencies=store_latencies,
            **kwargs,
        )

    return root_op.get_result()
