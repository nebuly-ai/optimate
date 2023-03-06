try:
    import neural_compressor  # noqa F401
    from neural_compressor.adaptor.pytorch import (
        _cfg_to_qconfig as cfg_to_qconfig,
        _cfgs_to_fx_cfgs as cfgs_to_fx_cfgs,
    )
    from neural_compressor.experimental import (
        MixedPrecision,
        Quantization,
        Pruning,
    )
except ImportError:
    cfg_to_qconfig = cfgs_to_fx_cfgs = None
    MixedPrecision = Quantization = Pruning = object
except ValueError:
    # MacOS
    cfg_to_qconfig = cfgs_to_fx_cfgs = None
    MixedPrecision = Quantization = Pruning = object
