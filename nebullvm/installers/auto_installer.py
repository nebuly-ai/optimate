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


SUPPORTED_BACKENDS = [
    "torch-full",
    "torch-base",
    "tensorflow-full",
    "tensorflow-base",
    "onnx-full",
    "onnx-base",
    "huggingface-full",
    "huggingface-full-tf",
    "huggingface-full-torch",
    "huggingface-base-tf",
    "huggingface-base-torch",
]

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


def check_backends(include_backends: Union[List[str], str]) -> List[str]:
    if isinstance(include_backends, str) and include_backends == "all":
        new_include_backends = list(INSTALLERS.keys())
    else:
        new_include_backends = []
        for backend in include_backends:
            if backend not in SUPPORTED_BACKENDS:
                raise ValueError(
                    f"Backend {backend} is not supported by nebullvm. "
                    f"Please check the docs to see all the supported options."
                )

            elif backend.startswith("torch"):
                new_include_backends.append("torch")
                if backend.endswith("full"):
                    new_include_backends.append("onnx")
            elif backend.startswith("tensorflow"):
                new_include_backends.append("tensorflow")
                if backend.endswith("full"):
                    new_include_backends.append("onnx")
            elif backend.startswith("huggingface"):
                new_include_backends.append("huggingface")
                if backend.endswith("full") or backend.endswith("tf"):
                    new_include_backends.append("tensorflow")
                if backend.endswith("full") or backend.endswith("torch"):
                    new_include_backends.append("torch")
                if "full" in backend:
                    new_include_backends.append("onnx")
            elif backend.startswith("onnx"):
                new_include_backends.append("onnx")

    # Remove duplicates
    new_include_backends = list(set(new_include_backends))
    new_include_backends.sort()

    return new_include_backends


def auto_install_libraries(
    include_backends: Union[List[str], str] = "all",
    include_compilers: Union[List[str], str] = "all",
):
    logger.info("Running auto install of nebullvm dependencies")

    include_backends = check_backends(include_backends)

    for backend in include_backends:
        backend_installer = INSTALLERS[backend](MODULES[backend])
        if not backend_installer.check_backend():
            backend_installer.install_backend()
        backend_installer.install_dependencies(include_backends)
        backend_installer.install_compilers(include_compilers)


def main():
    parser = argparse.ArgumentParser(
        description="Auto install dl backends and dependencies"
    )
    parser.add_argument(
        "-f",
        "--backends",
        help="The dl backends whose compilers will be installed",
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

    if len(args["backends"]) == 1 and args["backends"][0] == "all":
        backend_list = "all"
    else:
        backend_list = args["backends"]

    if len(args["compilers"]) == 1 and args["compilers"][0] == "all":
        compilers_list = "all"
    else:
        compilers_list = args["compilers"]

    auto_install_libraries(backend_list, compilers_list)


if __name__ == "__main__":
    main()
