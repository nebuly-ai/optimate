from nebullvm.optional_modules.dummy import DummyClass

try:
    import onnxsim
except ImportError:
    onnxsim = DummyClass
