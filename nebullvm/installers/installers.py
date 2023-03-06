import os
import platform
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import List

import cpuinfo
from loguru import logger

from nebullvm.config import (
    LIBRARIES_GPU,
)
from nebullvm.operations.optimizations.compilers.utils import (
    openvino_is_available,
    tensorrt_is_available,
    torch_tensorrt_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
)
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.utils import (
    gpu_is_available,
    check_module_version,
)


def get_cpu_arch():
    arch = cpuinfo.get_cpu_info()["arch"].lower()
    if "x86" in arch:
        return "x86"
    else:
        return "arm"


def _get_os():
    return platform.system()


def install_tvm(
    working_dir: str = None,
):
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
    if gpu_is_available():
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
        return True

    return True


def install_bladedisc():
    """Helper function for installing BladeDisc."""
    has_cuda = False
    if gpu_is_available():
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
    if not gpu_is_available():
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
        "https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.3.0",
    ]
    subprocess.run(cmd)

    try:
        import torch_tensorrt  # noqa F401
    except ImportError:
        return False

    return True


def install_tf2onnx():
    if _get_os() == "Darwin" and get_cpu_arch() == "arm":
        cmd = ["conda", "install", "-y", "tf2onnx>=1.8.4"]
        subprocess.run(cmd)
    else:
        cmd = ["pip3", "install", "--user", "protobuf<4,>=3.20.2"]
        subprocess.run(cmd)

        cmd = ["pip3", "install", "tf2onnx>=1.8.4"]
        subprocess.run(cmd)

    try:
        import tf2onnx  # noqa F401
    except ImportError:
        return False
    except AttributeError:
        # Sometimes the import could raise an attribute error
        # if installation fails
        pass

    return True


def install_tensor_rt():
    """Helper function for installing TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not gpu_is_available():
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

    openvino_version = "openvino-dev" if with_optimization else "openvino"
    cmd = ["pip3", "install", "--user", f"{openvino_version}>=2022.1.0"]
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
    if gpu_is_available():
        distribution_name = f"{distribution_name}-gpu"
    if _get_os() == "Darwin" and get_cpu_arch() == "arm":
        cmd = ["conda", "install", "-y", distribution_name]
    else:
        cmd = ["pip3", "install", distribution_name]
    subprocess.run(cmd)
    # install requirements for onnxruntime.transformers
    cmd = ["pip3", "install", "coloredlogs", "sympy"]
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
    if os_ in ["Darwin", "Windows"] or get_cpu_arch() == "arm":
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
        cmd = ["pip3", "install", "numpy>=1.22.0,<1.24.0"]
        subprocess.run(cmd)
    except Exception:
        # For python 3.7 numpy 1.22.0 is not available
        pass

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

    cmd = ["pip3", "install", "--user", "neural-compressor"]
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


class BaseInstaller(ABC):
    def __init__(self, module_list: List[str]):
        self.modules = module_list

    def install_compilers(
        self,
        include_libraries: List[str],
    ):
        for library in self.modules:
            if (
                isinstance(include_libraries, List)
                and library not in include_libraries
            ) or (not gpu_is_available() and library in LIBRARIES_GPU):
                continue

            logger.info(f"Trying to install {library} on the platform...")

            try:
                if not COMPILERS_AVAILABLE[library]():
                    install_ok = COMPILER_INSTALLERS[library]()
                else:
                    install_ok = True
            except Exception:
                install_ok = False

            if not install_ok:
                logger.warning(
                    f"Unable to install {library} on this platform. "
                    f"The compiler will be skipped. "
                )
            else:
                logger.info(f"{library} installed successfully!")

    @staticmethod
    def install_dependencies(include_framework: List[str]):
        raise NotImplementedError

    @staticmethod
    def check_framework():
        raise NotImplementedError

    @staticmethod
    def install_framework():
        raise NotImplementedError


class PytorchInstaller(BaseInstaller):
    @staticmethod
    def install_dependencies(include_framework: List[str]):
        return

    @staticmethod
    def check_framework():
        try:
            import torch  # noqa F401
        except ImportError:
            raise ImportError(
                "No PyTorch found in your python environment. Please install "
                "it from https://pytorch.org/get-started/locally/."
            )

        if not check_module_version(torch, min_version="1.12.0"):
            logger.warning(
                "You are using an old version of PyTorch, please update it "
                "in order to get the best optimization results."
            )

        return True

    @staticmethod
    def install_framework():
        cmd = ["pip3", "install", "torch>=1.10.0"]
        subprocess.run(cmd)

        try:
            import torch  # noqa F401
        except ImportError:
            return False

        return True


class TensorflowInstaller(BaseInstaller):
    @staticmethod
    def install_dependencies(include_framework: List[str]):
        if "onnx" in include_framework:
            install_tf2onnx()

    @staticmethod
    def check_framework():
        try:
            import tensorflow  # noqa F401
        except ImportError:
            return False

        if not check_module_version(tensorflow, min_version="2.7.0"):
            return False

        return True

    @staticmethod
    def install_framework():
        if _get_os() == "Darwin" and get_cpu_arch() == "arm":
            cmd = ["conda", "install", "-y", "tensorflow>=2.7.0", "numpy<1.24"]
            subprocess.run(cmd)
        else:
            cmd = ["pip3", "install", "--user", "tensorflow>=2.7.0"]
            subprocess.run(cmd)

        try:
            import tensorflow  # noqa F401
        except ImportError:
            return False

        return True


class ONNXInstaller(BaseInstaller):
    @staticmethod
    def install_dependencies(include_framework: List[str]):
        install_onnxruntime()
        cmd = ["pip3", "install", "onnxmltools>=1.11.0"]
        subprocess.run(cmd)
        install_onnx_simplifier()

    @staticmethod
    def check_framework():
        try:
            import onnx  # noqa F401
        except ImportError:
            return False

        if not check_module_version(onnx, min_version="1.10.0"):
            return False

        return True

    @staticmethod
    def install_framework():
        if _get_os() == "Darwin" and get_cpu_arch() == "arm":
            cmd = ["pip3", "install", "cmake"]
            subprocess.run(cmd)

        cmd = ["pip3", "install", "onnx>=1.10.0"]
        subprocess.run(cmd)

        try:
            import onnx  # noqa F401
        except ImportError:
            return False

        return True


class HuggingFaceInstaller(BaseInstaller):
    @staticmethod
    def install_dependencies(include_framework: List[str]):
        pass

    @staticmethod
    def check_framework():
        try:
            import transformers  # noqa F401
        except ImportError:
            return False

        return True

    @staticmethod
    def install_framework():
        cmd = ["pip3", "install", "transformers"]
        subprocess.run(cmd)

        try:
            import transformers  # noqa F401
        except ImportError:
            return False

        return True


COMPILER_INSTALLERS = {
    "openvino": install_openvino,
    "tensor_rt": install_tensor_rt,
    "torch_tensor_rt": install_torch_tensor_rt,
    "deepsparse": install_deepsparse,
    "intel_neural_compressor": install_intel_neural_compressor,
}

COMPILERS_AVAILABLE = {
    "openvino": openvino_is_available,
    "tensor_rt": tensorrt_is_available,
    "torch_tensor_rt": torch_tensorrt_is_available,
    "deepsparse": deepsparse_is_available,
    "intel_neural_compressor": intel_neural_compressor_is_available,
}
