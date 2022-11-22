try:
    import torch_tensorrt
    from torch_tensorrt import Calibrator
except ImportError:
    torch_tensorrt = object
    Calibrator = None
