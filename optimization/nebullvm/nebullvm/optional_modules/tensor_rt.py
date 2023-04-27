from nebullvm.optional_modules.dummy import DummyClass

try:
    import tensorrt
    from tensorrt import IInt8EntropyCalibrator2
except ImportError:
    tensorrt = DummyClass
    IInt8EntropyCalibrator2 = DummyClass

try:
    import polygraphy.cuda as polygraphy
    from polygraphy.logger import G_LOGGER

    G_LOGGER.module_severity = 40
    from polygraphy.backend.onnx.loader import fold_constants
except ImportError:
    polygraphy = DummyClass
    fold_constants = DummyClass
