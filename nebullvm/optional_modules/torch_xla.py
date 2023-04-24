from nebullvm.optional_modules.dummy import DummyClass

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    torch_xla = DummyClass
    xm = DummyClass
