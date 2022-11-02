import logging

logger = logging.getLogger("nebullvm_logger")

try:
    import tensorrt
    from tensorrt import IInt8EntropyCalibrator2
except ImportError:
    logger.warning(
        "Missing Library: "
        "tensorrt module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    tensorrt = object
    IInt8EntropyCalibrator2 = object

try:
    import polygraphy.cuda as polygraphy
except ImportError:
    logger.warning(
        "Missing Library: "
        "polygraphy module is not installed on this platform. "
        "It's needed for tensorrt to work properly, please install "
        "it if you want to include tensorrt in the optimization "
        "pipeline."
    )
    polygraphy = object
