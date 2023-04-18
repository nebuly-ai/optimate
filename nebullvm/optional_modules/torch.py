from nebullvm.optional_modules.dummy import DummyClass

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
    import torch.distributed as torch_distributed
except ImportError:

    class nn:
        Module = DummyClass

    class jit:
        ScriptModule = DummyClass

    class fx:
        GraphModule = DummyClass

    class torch:
        float = half = int8 = DummyClass
        float16 = float32 = int32 = int64 = DummyClass
        Tensor = DummyClass
        dtype = DummyClass
        nn = nn
        jit = jit
        Generator = DummyClass
        FloatTensor = DummyClass
        fx = fx

        @staticmethod
        def no_grad():
            return lambda x: None

        @staticmethod
        def inference_mode():
            return lambda x: None

    Dataset = DummyClass
    Module = DummyClass
    ScriptModule = DummyClass
    GraphModule = DummyClass
    DataLoader = DummyClass
    symbolic_trace = None
    QuantStub = DeQuantStub = DummyClass
    default_dynamic_qconfig = prepare_fx = convert_fx = None
    Generator = DummyClass
    FloatTensor = DummyClass
    torch_distributed = None
