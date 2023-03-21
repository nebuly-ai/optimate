from nebullvm.optional_modules.dummy import DummyClass

try:
    from deepsparse import compile_model, cpu
except ImportError:
    compile_model = cpu = DummyClass
