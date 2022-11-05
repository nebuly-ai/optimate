import logging
import os
import platform
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Optional, List

import cpuinfo
import torch

from nebullvm.base import DeepLearningFramework
from nebullvm.config import (
    LIBRARIES_GPU,
    ONNX_MODULES,
    TORCH_MODULES,
    TENSORFLOW_MODULES,
)
from nebullvm.utils.compilers import (
    openvino_is_available,
    tensorrt_is_available,
    torch_tensorrt_is_available,
    deepsparse_is_available,
    intel_neural_compressor_is_available,
)
from nebullvm.utils.general import (
    check_module_version,
    is_python_version_3_10,
    gpu_is_available,
)

logger = logging.getLogger("nebullvm_logger")


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
    dl_framework: DeepLearningFramework = DeepLearningFramework.PYTORCH,
):
    """Helper function for installing ApacheTVM.

    This function needs some prerequisites for running, as a valid `git`
    installation and having MacOS or a Linux-distribution as OS.

    Args:
        working_dir (str, optional): The directory where the tvm repo will be
            cloned and installed.
        dl_framework: (DeepLearningFramework, optional): Deep learning
            framework.
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
    if gpu_is_available(dl_framework):
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


def install_bladedisc(
    dl_framework: DeepLearningFramework = DeepLearningFramework.PYTORCH,
):
    """Helper function for installing BladeDisc."""
    has_cuda = False
    if gpu_is_available(dl_framework):
        has_cuda = True

    path = Path(__file__).parent
    installation_file = str(path / "install_bladedisc.sh")
    subprocess.Popen(["bash", installation_file, str(has_cuda).lower()])

    try:
        import torch_blade  # noqa F401
    except ImportError:
        return False

    return True


def install_torch_tensor_rt(
    dl_framework: DeepLearningFramework = DeepLearningFramework.PYTORCH,
):
    """Helper function for installing Torch-TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not gpu_is_available(dl_framework):
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
    if _get_os() == "Darwin" and get_cpu_arch() == "arm":
        cmd = ["conda", "install", "-y", "tf2onnx>=1.8.4"]
        subprocess.run(cmd)
    else:
        cmd = ["pip3", "install", "tf2onnx>=1.8.4"]
        subprocess.run(cmd)

    try:
        import tf2onnx  # noqa F401
    except ImportError:
        return False

    return True


def install_tensor_rt(
    dl_framework: DeepLearningFramework = DeepLearningFramework.PYTORCH,
):
    """Helper function for installing TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not gpu_is_available(dl_framework):
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


def install_openvino(
    with_optimization: bool = True, dl_framework: DeepLearningFramework = None
):
    """Helper function for installing the OpenVino compiler.

    This function just works on intel machines.

    Args:
        with_optimization (bool): Flag for installing the full openvino engine
            or limiting the installation to the tools need for inference
            models.
        dl_framework: (DeepLearningFramework, optional): Deep learning
            framework.
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


def install_onnxruntime(
    dl_framework: DeepLearningFramework = DeepLearningFramework.NUMPY,
):
    """Helper function for installing the right version of onnxruntime."""
    distribution_name = "onnxruntime"
    if gpu_is_available(dl_framework):
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


def install_deepsparse(dl_framework: DeepLearningFramework = None):
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


def install_intel_neural_compressor(
    dl_framework: DeepLearningFramework = None,
):
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


class BaseInstaller(ABC):
    def __init__(self, module_list: List[str]):
        self.modules = module_list

    def install_compilers(
        self,
        dl_framework: DeepLearningFramework,
        include_libraries: Optional[List[str]] = None,
    ):
        for library in self.modules:
            if (
                isinstance(include_libraries, List)
                and library not in include_libraries
            ) or (
                not gpu_is_available(dl_framework) and library in LIBRARIES_GPU
            ):
                continue

            logger.info(f"Trying to install {library} on the platform...")

            try:
                if not COMPILERS_AVAILABLE[library]():
                    install_ok = COMPILER_INSTALLERS[library](
                        dl_framework=dl_framework
                    )
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
    def install_dependencies(include_frameworks: List[str]):
        raise NotImplementedError

    @staticmethod
    def check_framework():
        raise NotImplementedError

    @staticmethod
    def install_framework():
        raise NotImplementedError


class PytorchInstaller(BaseInstaller, ABC):
    @staticmethod
    def install_dependencies(include_frameworks: List[str]):
        return

    @staticmethod
    def check_framework():
        try:
            import torch  # noqa F401
        except ImportError:
            return False

        if not check_module_version(torch, min_version="1.12.0"):
            return False

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


class TensorflowInstaller(BaseInstaller, ABC):
    @staticmethod
    def install_dependencies(include_frameworks: List[str]):
        if "onnx" in include_frameworks:
            install_tf2onnx()

    @staticmethod
    def check_framework():
        try:
            import tensorflow  # noqa F401
        except ImportError:
            return False

        if not check_module_version(
            tensorflow, min_version="2.7.0", max_version="2.10.0"
        ):
            return False

        return True

    @staticmethod
    def install_framework():
        # Tensorflow 2.10 for now it's not supported
        # tf2onnx library does not support flatbuffers >= 2.x
        if _get_os() == "Darwin" and get_cpu_arch() == "arm":
            cmd = ["conda", "install", "-y", "tensorflow>=2.7.0,<2.10"]
            subprocess.run(cmd)
        else:
            cmd = ["pip3", "install", "tensorflow>=2.7.0,<2.10"]
            subprocess.run(cmd)

        try:
            import tensorflow  # noqa F401
        except ImportError:
            return False

        return True


class ONNXInstaller(BaseInstaller, ABC):
    @staticmethod
    def install_dependencies(include_frameworks: List[str]):
        cmd = ["pip3", "install", "onnxmltools>=1.11.0"]
        subprocess.run(cmd)
        install_onnx_simplifier()
        install_onnxruntime()

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


def auto_install_libraries(
    include_frameworks: List, include_compilers: Optional[List[str]] = None
):
    logger.info("Running auto install of nebullvm dependencies")

    for framework in include_frameworks:
        framework_installer = INSTALLERS[framework](MODULES[framework])
        if not framework_installer.check_framework():
            framework_installer.install_framework()
        framework_installer.install_dependencies(include_frameworks)
        framework_installer.install_compilers(
            FRAMEWORKS[framework], include_compilers
        )


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


INSTALLERS = {
    "onnx": ONNXInstaller,
    "torch": PytorchInstaller,
    "tensorflow": TensorflowInstaller,
}

MODULES = {
    "onnx": ONNX_MODULES,
    "torch": TORCH_MODULES,
    "tensorflow": TENSORFLOW_MODULES,
}

FRAMEWORKS = {
    "torch": DeepLearningFramework.PYTORCH,
    "tensorflow": DeepLearningFramework.TENSORFLOW,
    "onnx": DeepLearningFramework.NUMPY,
}
