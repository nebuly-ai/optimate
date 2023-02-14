import cpuinfo
from loguru import logger

from nebullvm.operations.optimizations.compilers.utils import (
    tvm_is_available,
    onnxruntime_is_available,
    tensorrt_is_available,
    openvino_is_available,
    bladedisc_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
    torch_tensorrt_is_available,
)
from nebullvm.tools.base import Device, DeviceType
from nebullvm.tools.utils import gpu_is_available, check_module_version


def torch_is_available() -> bool:
    try:
        import torch  # noqa F401

        if not torch.cuda.is_available() and gpu_is_available():
            logger.warning(
                "Installed PyTorch does not have cuda support. "
                "Please ensure that torch.cuda.is_available() "
                "returns True by installing the proper version "
                "of PyTorch. "
            )

        if not check_module_version(torch, min_version="1.10.0"):
            logger.warning(
                "torch module version must be >= 1.10.0. "
                "Please update it if you want to use it."
            )
            return False
    except ImportError:
        return False
    else:
        return True


def tensorflow_is_available() -> bool:
    try:
        import tensorflow  # noqa F401

        if not check_module_version(tensorflow, min_version="2.7.0"):
            logger.warning(
                "tensorflow module version must be >= 2.7.0. "
                "Please update it if you want to use it."
            )
            return False
    except ImportError:
        return False
    else:
        return True


def onnx_is_available() -> bool:
    try:
        import onnx  # noqa F401

        if not check_module_version(onnx, min_version="1.10.0"):
            logger.warning(
                "onnx module version must be >= 1.10.0. "
                "Please update it if you want to use it."
            )
            return False
        return True
    except ImportError:
        return False


def _onnxmltools_is_available():
    try:
        import onnxmltools  # noqa F401

        if not check_module_version(onnxmltools, min_version="1.11.0"):
            logger.warning(
                "onnxmltools module version must be >= 1.11.0. "
                "Please update it if you want to use the ONNX API "
                "or the ONNX pipeline for PyTorch and Tensorflow."
            )
            return False
        else:
            return True
    except ImportError:
        return False


def _onnxsim_is_available():
    try:
        import onnxsim  # noqa F401

        return True
    except ImportError:
        return False


def _polygraphy_is_available():
    try:
        import polygraphy.cuda  # noqa F401

        return True
    except ImportError:
        return False


def tf2onnx_is_available():
    try:
        import tf2onnx  # noqa F401

        return True
    except ImportError:
        return False


def check_dependencies(device: Device):
    missing_frameworks = []
    missing_suggested_compilers = []
    missing_optional_compilers = []
    missing_dependencies = []

    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()

    if not onnx_is_available():
        missing_frameworks.append("onnx")

    if not tvm_is_available():
        missing_optional_compilers.append("tvm")
    if not onnxruntime_is_available():
        missing_suggested_compilers.append("onnxruntime")
    elif not _onnxmltools_is_available():
        missing_dependencies.append("onnxmltools")
    if device.type is DeviceType.GPU:
        if not tensorrt_is_available():
            missing_suggested_compilers.append("tensorrt")
        else:
            if not _onnxsim_is_available():
                missing_dependencies.append("onnxsim")
            elif not _polygraphy_is_available():
                missing_dependencies.append("polygraphy")
    if device.type is DeviceType.CPU:
        if not openvino_is_available() and "intel" in processor:
            missing_suggested_compilers.append("openvino")

    if torch_is_available():
        if not tvm_is_available():
            if "tvm" not in missing_optional_compilers:
                missing_optional_compilers.append("tvm")
        if not bladedisc_is_available():
            missing_optional_compilers.append("torch_blade")

        if device.type is DeviceType.CPU:
            if not deepsparse_is_available() and "intel" in processor:
                missing_suggested_compilers.append("deepsparse")
            if (
                not intel_neural_compressor_is_available()
                and "intel" in processor
            ):
                missing_suggested_compilers.append("neural_compressor")
        elif device.type is DeviceType.GPU:
            if not torch_tensorrt_is_available:
                missing_suggested_compilers.append("torch_tensorrt")
    else:
        missing_frameworks.append("torch")

    if tensorflow_is_available():
        if not tf2onnx_is_available():
            missing_dependencies.append("tf2onnx")
    else:
        missing_frameworks.append("tensorflow")

    missing_frameworks = ", ".join(missing_frameworks)
    if len(missing_frameworks) > 0:
        logger.warning(
            f"Missing Frameworks: {missing_frameworks}.\n "
            f"Please install them "
            "to include them in the optimization pipeline."
        )

    missing_suggested_compilers = ", ".join(missing_suggested_compilers)
    if len(missing_suggested_compilers) > 0:
        logger.warning(
            f"Missing Compilers: {missing_suggested_compilers}.\n "
            f"Please install them "
            "to include them in the optimization pipeline."
        )

    missing_dependencies = ", ".join(missing_dependencies)
    if len(missing_dependencies) > 0:
        logger.warning(
            f"Missing Dependencies: {missing_dependencies}.\n "
            f"Without them, some compilers "
            f"may not work properly."
        )
