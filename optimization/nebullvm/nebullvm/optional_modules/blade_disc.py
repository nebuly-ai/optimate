from nebullvm.optional_modules.dummy import DummyClass

try:
    import torch_blade
except ImportError:
    torch_blade = DummyClass
