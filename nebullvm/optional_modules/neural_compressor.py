import logging

logger = logging.getLogger("nebullvm_logger")

try:
    from neural_compressor.adaptor.pytorch import (
        _cfg_to_qconfig,
        _cfgs_to_fx_cfgs,
    )
    from neural_compressor.experimental import (
        MixedPrecision,
        Quantization,
        Pruning,
    )
except ImportError:
    logger.warning(
        "Missing Library: "
        "neural_compressor module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    _cfg_to_qconfig = _cfgs_to_fx_cfgs = None
    MixedPrecision = Quantization = Pruning = object
except ValueError:
    # MacOS
    MixedPrecision = Quantization = Pruning = object
