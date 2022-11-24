from typing import Any, Union, Iterable, Sequence, Callable, Dict, List

from nebullvm.operations.root.black_box import BlackBoxModelOptimizationRootOp


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
    device: str = "CPU",
):
    root_op = BlackBoxModelOptimizationRootOp()
    root_op.to(device).execute(
        model,
        input_data,
        metric_drop_ths,
        metric,
        optimization_time,
        dynamic_info,
        config_file,
        ignore_compilers,
        ignore_compressors,
        store_latencies,
    )

    return root_op.optimal_model
