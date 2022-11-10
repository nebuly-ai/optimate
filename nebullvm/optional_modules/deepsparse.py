try:
    from deepsparse import compile_model, cpu
except ImportError:
    compile_model = cpu = object
