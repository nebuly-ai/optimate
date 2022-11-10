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
    tvm = (
        IRModule
    ) = (
        NDArray
    ) = (
        XGBTuner
    ) = autotvm = relay = ToMixedPrecision = GraphModule = Module = object
