import warnings

try:
    import torch_tensorrt
except ImportError:
    warnings.warn(
        "torch_tensorrt module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    torch_tensorrt = object
