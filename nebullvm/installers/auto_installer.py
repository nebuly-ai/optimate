import argparse
from typing import List, Union

from loguru import logger

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


SUPPORTED_BACKENDS_DICT = {
    "torch": ["onnx"],
    "tensorflow": ["onnx"],
    "huggingface": ["torch", "tensorflow", "onnx"],
    "onnx": [],
}

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


def select_frameworks_to_install(
    include_frameworks: Union[List[str], str],
    include_backends: Union[List[str], str],
) -> List[str]:
    supported_frameworks = list(INSTALLERS.keys())
    if isinstance(include_frameworks, str) and include_frameworks == "all":
        frameworks_list = supported_frameworks
    elif isinstance(include_frameworks, list):
        frameworks_list = []
        for framework in include_frameworks:
            if framework in supported_frameworks:
                frameworks_list.append(framework)
            else:
                logger.warning(f"Framework {framework} not supported")

        if len(frameworks_list) == 0:
            raise ValueError("No supported frameworks selected")

        if isinstance(include_backends, str) and include_backends == "all":
            for framework in frameworks_list:
                for backend in SUPPORTED_BACKENDS_DICT[framework]:
                    frameworks_list.append(backend)
        elif isinstance(include_backends, list):
            for backend in include_backends:
                if backend not in supported_frameworks:
                    logger.warning(f"Backend {backend} not supported")
                else:
                    backend_supported = False
                    for framework in frameworks_list:
                        if backend in SUPPORTED_BACKENDS_DICT[framework]:
                            frameworks_list.append(backend)
                            backend_supported = True
                            break
                    if not backend_supported:
                        logger.warning(
                            f"Backend {backend} not supported for selected "
                            f"frameworks"
                        )
        else:
            raise ValueError("Invalid backends list")
    else:
        raise ValueError("Invalid frameworks list")

    frameworks_list = list(set(frameworks_list))
    frameworks_list.sort()

    return frameworks_list


def select_compilers_to_install(
    include_compilers: Union[List[str], str], framework_list: List[str]
) -> List[str]:
    compiler_list = []
    supported_compilers = list(
        set([item for sublist in MODULES.values() for item in sublist])
    )
    if isinstance(include_compilers, str) and include_compilers == "all":
        compiler_list = list(
            set(
                [
                    item
                    for (fr, compilers) in MODULES.items()
                    for item in compilers
                    if fr in framework_list
                ]
            )
        )
    else:
        for compiler in include_compilers:
            if compiler not in supported_compilers:
                logger.warning(f"Compiler {compiler} not supported")
            else:
                compiler_supported = False
                for framework in framework_list:
                    if compiler in MODULES[framework]:
                        compiler_list.append(compiler)
                        compiler_supported = True
                        break
                if not compiler_supported:
                    logger.warning(
                        f"Compiler {compiler} not supported for selected "
                        f"frameworks"
                    )

    compiler_list = list(set(compiler_list))
    compiler_list.sort()

    return compiler_list


def auto_install_libraries(
    include_frameworks: Union[List[str], str] = "all",
    include_backends: Union[List[str], str] = "all",
    include_compilers: Union[List[str], str] = "all",
):
    logger.info("Running auto install of nebullvm dependencies")

    framework_list = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    compilers_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    for framework in framework_list:
        framework_installer = INSTALLERS[framework](MODULES[framework])
        if not framework_installer.check_framework():
            framework_installer.install_framework()
        framework_installer.install_dependencies(framework_list)
        framework_installer.install_compilers(compilers_list)


def main():
    parser = argparse.ArgumentParser(
        description="Auto install dl frameworks and dependencies"
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        help="The base dl frameworks to be installed",
        default="all",
        nargs="+",
    )
    parser.add_argument(
        "-b",
        "--extra-backends",
        help="additional dl frameworks to be installed to "
        "gain the optimal speedup",
        default="all",
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

    if len(args["frameworks"]) == 1 and args["frameworks"][0] == "all":
        framework_list = "all"
    else:
        framework_list = args["frameworks"]

    if len(args["extra_backends"]) == 1 and args["extra_backends"][0] in [
        "all",
        "none",
    ]:
        if args["extra_backends"][0] == "all":
            backend_list = "all"
        else:
            backend_list = []
    else:
        backend_list = args["extra_backends"]

    if len(args["compilers"]) == 1 and args["compilers"][0] == "all":
        compilers_list = "all"
    else:
        compilers_list = args["compilers"]

    auto_install_libraries(framework_list, backend_list, compilers_list)


if __name__ == "__main__":
    main()
