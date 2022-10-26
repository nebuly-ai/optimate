import warnings

try:
    import onnxsim
except ImportError:
    warnings.warn(
        "onnxsim module is not installed on this platform. "
        "It's an optional requirement of tensorrt. "
        "Installing it could solve some issues with transformers. "
    )
    onnxsim = object
