import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional, List

import cpuinfo
import torch

from nebullvm.utils.general import check_module_version, is_python_version_3_10


def get_cpu_arch():
    arch = cpuinfo.get_cpu_info()["arch"].lower()
    if "x86" in arch:
        return "x86"
    else:
        return "arm"


def _get_os():
    return platform.system()


def install_tvm(working_dir: str = None):
    """Helper function for installing ApacheTVM.

    This function needs some prerequisites for running, as a valid `git`
    installation and having MacOS or a Linux-distribution as OS.

    Args:
        working_dir (str, optional): The directory where the tvm repo will be
            cloned and installed.
    """
    path = Path(__file__).parent
    # install pre-requisites
    installation_file_prerequisites = str(
        path / "install_tvm_prerequisites.sh"
    )
    subprocess.run(
        ["bash", installation_file_prerequisites],
        cwd=working_dir or Path.home(),
    )
    installation_file = str(path / "install_tvm.sh")
    hardware_config = get_cpu_arch()
    if torch.cuda.is_available():
        hardware_config = f"{hardware_config}_cuda"
    env_dict = {
        "CONFIG_PATH": str(
            path / f"tvm_installers/{hardware_config}/config.cmake"
        ),
        **dict(os.environ.copy()),
    }
    subprocess.run(
        ["bash", installation_file],
        cwd=working_dir or Path.home(),
        env=env_dict,
    )

    try:
        import tvm  # noqa F401
    except ImportError:
        return False

    return True


def install_bladedisc():
    """Helper function for installing BladeDisc."""
    has_cuda = False
    if torch.cuda.is_available():
        has_cuda = True

    path = Path(__file__).parent
    installation_file = str(path / "install_bladedisc.sh")
    subprocess.Popen(["bash", installation_file, str(has_cuda).lower()])

    try:
        import torch_blade  # noqa F401
    except ImportError:
        return False

    return True


def install_torch_tensor_rt():
    """Helper function for installing Torch-TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Torch-TensorRT can run just on Nvidia machines. "
            "No available cuda driver has been found."
        )
    elif not check_module_version(torch, min_version="1.12.0"):
        raise RuntimeError(
            "Torch-TensorRT can be installed only from Pytorch 1.12. "
            "Please update your Pytorch version."
        )

    # Verify that TensorRT is installed, otherwise install it
    try:
        import tensorrt  # noqa F401
    except ImportError:
        install_tensor_rt()

    # cmd = [
    #     "pip3",
    #     "install",
    #     "torch-tensorrt>=1.2.0",
    #     "-f",
    #     "https://github.com/pytorch/TensorRT/releases",
    # ]

    cmd = [
        "pip3",
        "install",
        "torch-tensorrt",
        "--find-links",
        "https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.2.0",
    ]
    subprocess.run(cmd)

    try:
        import torch_tensorrt  # noqa F401
    except ImportError:
        return False

    return True


def install_tf2onnx():
    cmd = ["pip3", "install", "tf2onnx>=1.8.4"]
    subprocess.run(cmd)

    try:
        import tf2onnx  # noqa F401
    except ImportError:
        return False

    return True


def install_tensorflow():
    # Tensorflow 2.10 for now it's not supported
    # Will be supported when tf2onnx library will support flatbuffers >= 2.x
    cmd = ["pip3", "install", "tensorflow>=2.7.0,<2.10"]
    subprocess.run(cmd)

    try:
        import tensorflow  # noqa F401
    except ImportError:
        return False

    return True


def install_tensor_rt():
    """Helper function for installing TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT can run just on Nvidia machines. "
            "No available cuda driver has been found."
        )
    path = Path(__file__).parent
    installation_file = str(path / "install_tensor_rt.sh")
    subprocess.run(["bash", installation_file])

    try:
        import polygraphy  # noqa F401
        import tensorrt  # noqa F401
    except ImportError:
        return False

    return True


def install_openvino(with_optimization: bool = True):
    """Helper function for installing the OpenVino compiler.

    This function just works on intel machines.

    Args:
        with_optimization (bool): Flag for installing the full openvino engine
            or limiting the installation to the tools need for inference
            models.
    """
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        raise RuntimeError(
            f"Openvino can run just on Intel machines. "
            f"You are trying to install it on {processor}"
        )
    if is_python_version_3_10():
        raise RuntimeError(
            "Openvino does not support python 3.10. " "It won't be installed."
        )

    openvino_version = "openvino-dev" if with_optimization else "openvino"
    cmd = ["pip3", "install", f"{openvino_version}[onnx]>=2022.1.0"]
    subprocess.run(cmd)

    # Reinstall updated versions of libraries that were downgraded by openvino
    cmd = ["pip3", "install", "onnx>=1.12"]
    subprocess.run(cmd)

    cmd = ["pip3", "install", "scipy>=1.7.3"]
    subprocess.run(cmd)

    try:
        from openvino.runtime import (  # noqa F401
            Core,
            Model,
            CompiledModel,
            InferRequest,
        )
    except ImportError:
        return False

    return True


def install_onnxruntime():
    """Helper function for installing the right version of onnxruntime."""
    distribution_name = "onnxruntime"
    if torch.cuda.is_available():
        distribution_name = f"{distribution_name}-gpu"
    if _get_os() == "Darwin" and get_cpu_arch() == "arm":
        cmd = ["conda", "install", "-y", distribution_name]
    else:
        cmd = ["pip3", "install", distribution_name]
    subprocess.run(cmd)
    # install requirements for onnxruntime.transformers
    cmd = ["pip3", "install", "coloredlogs", "sympy", "transformers"]
    subprocess.run(cmd)

    try:
        import onnxruntime  # noqa F401
    except ImportError:
        return False

    return True


def install_deepsparse():
    """Helper function for installing DeepSparse."""
    python_minor_version = sys.version_info.minor

    os_ = platform.system()
    if os_ == "Darwin" or get_cpu_arch() == "arm":
        raise RuntimeError(
            "DeepSparse is not supported on this platform. "
            "It won't be installed."
        )

    try:
        cmd = ["apt-get", "install", f"python3.{python_minor_version}-venv"]
        subprocess.run(cmd)
    except Exception:
        pass

    cmd = ["pip3", "install", "deepsparse"]
    subprocess.run(cmd)

    try:
        from deepsparse import compile_model, cpu  # noqa F401
    except ImportError:
        return False

    return True


def install_intel_neural_compressor():
    """Helper function for installing Intel Neural Compressor."""

    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        raise RuntimeError(
            f"Intel Neural Compressor can run just on Intel machines. "
            f"You are trying to install it on {processor}"
        )

    cmd = ["pip3", "install", "neural-compressor"]
    subprocess.run(cmd)

    try:
        from neural_compressor.experimental import (  # noqa F401
            MixedPrecision,
            Quantization,
        )
    except ImportError:
        return False

    return True


def install_onnx_simplifier():
    """Helper function for installing ONNX simplifier."""

    if get_cpu_arch() != "arm":
        # Install onnx simplifier
        cmd = ["pip3", "install", "onnxsim"]
        subprocess.run(cmd)

    try:
        import onnxsim  # noqa F401
    except ImportError:
        return False

    return True


def auto_install_libraries(include_libraries: Optional[List[str]] = None):
    try:
        import tensorflow  # noqa F401

        install_tf2onnx()
    except ImportError:
        pass

    for library in [
        "onnxruntime",
        "openvino",
        "deepsparse",
        "intel_neural_compressor",
    ]:
        if (
            isinstance(include_libraries, List)
            and library not in include_libraries
        ):
            continue

        try:
            install_ok = COMPILER_INSTALLERS[library]()
        except Exception:
            install_ok = False

        if not install_ok:
            warnings.warn(
                f"Unable to install {library} on this platform. "
                f"The compiler will be skipped. "
            )

    if torch.cuda.is_available():
        for library in ["tensor_rt", "torch_tensor_rt", "onnx_simplifier"]:
            if (
                isinstance(include_libraries, List)
                and library not in include_libraries
            ):
                continue

            try:
                install_ok = COMPILER_INSTALLERS[library]()
            except Exception:
                install_ok = False

            if not install_ok:
                warnings.warn(
                    f"Unable to install {library} on this platform. "
                    f"The compiler will be skipped. "
                )


COMPILER_INSTALLERS = {
    "onnxruntime": install_onnxruntime,
    "openvino": install_openvino,
    "tensor_rt": install_tensor_rt,
    "torch_tensor_rt": install_torch_tensor_rt,
    "deepsparse": install_deepsparse,
    "intel_neural_compressor": install_intel_neural_compressor,
    "onnx_simplifier": install_onnx_simplifier,
}
