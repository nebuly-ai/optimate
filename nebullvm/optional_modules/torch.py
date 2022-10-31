import logging

from nebullvm.utils.general import check_module_version

logger = logging.getLogger("nebullvm_logger")

try:
    import torch  # noqa F401
    from torch.nn import Module  # noqa F401
    from torch.jit import ScriptModule  # noqa F401
    from torch.fx import GraphModule
    from torch.utils.data import DataLoader, Dataset  # noqa F401
    from torch.quantization.quantize_fx import (  # noqa F401
        prepare_fx,
        convert_fx,
    )

    from torch.ao.quantization.stubs import QuantStub, DeQuantStub
    from torch.fx import symbolic_trace
    from torch.quantization import default_dynamic_qconfig

    if not check_module_version(torch, min_version="1.10.0"):
        logger.warning(
            "torch module version must be >= 1.10.0. "
            "Please update it if you want to use it."
        )
except ImportError:
    logger.warning(
        "Missing Library: "
        "torch module is not installed on this platform. "
        "Please install it if you want to use torch API."
    )

    class Tensor:
        pass

    class torch:
        float = half = int8 = None
        Tensor = Tensor
        dtype = None

        @staticmethod
        def no_grad():
            return lambda x: None

    class Module:
        pass

    class ScriptModule:
        pass

    class GraphModule:
        pass

    Dataset = DataLoader = object
    symbolic_trace = None
    QuantStub = (
        DeQuantStub
    ) = default_dynamic_qconfig = prepare_fx = convert_fx = None
