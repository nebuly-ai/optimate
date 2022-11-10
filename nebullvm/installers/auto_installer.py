import argparse
import logging
from typing import List, Union

from nebullvm.config import (
    ONNX_MODULES,
    TENSORFLOW_MODULES,
    TORCH_MODULES,
    HUGGING_FACE_MODULES,
)
from nebullvm.installers.installers import (
    ONNXInstaller,
    PytorchInstaller,
    TensorflowInstaller,
    HuggingFaceInstaller,
)


logger = logging.getLogger("nebullvm_logger")

INSTALLERS = {
    "onnx": ONNXInstaller,
    "torch": PytorchInstaller,
    "tensorflow": TensorflowInstaller,
    "huggingface": HuggingFaceInstaller,
}

MODULES = {
    "onnx": ONNX_MODULES,
    "torch": TORCH_MODULES,
    "tensorflow": TENSORFLOW_MODULES,
    "huggingface": HUGGING_FACE_MODULES,
}


def auto_install_libraries(
    include_frameworks: List[str],
    include_compilers: Union[List[str], str] = "all",
):
    logger.info("Running auto install of nebullvm dependencies")

    for framework in include_frameworks:
        framework_installer = INSTALLERS[framework](MODULES[framework])
        if not framework_installer.check_framework():
            framework_installer.install_framework()
        framework_installer.install_dependencies(include_frameworks)
        framework_installer.install_compilers(include_compilers)


def main():
    parser = argparse.ArgumentParser(
        description="Auto install dl frameworks and dependencies"
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        help="The dl frameworks whose compilers will be installed",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-c",
        "--compilers",
        help="Compilers to be installed",
        default="all",
        nargs="+",
    )
    args = vars(parser.parse_args())

    framework_list = args["frameworks"]
    if len(args["compilers"]) == 1 and args["compilers"][0] == "all":
        compilers_list = "all"
    else:
        compilers_list = args["compilers"]

    auto_install_libraries(framework_list, compilers_list)


if __name__ == "__main__":
    main()
