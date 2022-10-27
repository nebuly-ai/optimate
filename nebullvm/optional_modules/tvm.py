import logging

logger = logging.getLogger(__name__)

try:
    import tvm
    from tvm import IRModule
    from tvm.runtime.ndarray import NDArray
    from tvm.autotvm.tuner import XGBTuner
    from tvm import autotvm
    import tvm.relay as relay
    from tvm.relay.transform import ToMixedPrecision
    from tvm.contrib.graph_executor import GraphModule
    from tvm.runtime import Module
except ImportError:
    logger.warning(
        "tvm module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )

    tvm = (
        IRModule
    ) = (
        NDArray
    ) = (
        XGBTuner
    ) = autotvm = relay = ToMixedPrecision = GraphModule = Module = object
