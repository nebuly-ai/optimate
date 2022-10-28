import logging
import os
import sys
import warnings
from packaging import version
from types import ModuleType


def check_module_version(
    module: ModuleType, min_version: str = None, max_version: str = None
) -> bool:
    installed_version = module.__version__

    if min_version is not None:
        if version.parse(installed_version) < version.parse(min_version):
            return False

    if max_version is not None:
        if version.parse(installed_version) > version.parse(max_version):
            return False

    return True


def is_python_version_3_10():
    return (
        str(sys.version_info.major) + "." + str(sys.version_info.minor)
        == "3.10"
    )


def debug_mode_enabled():
    return int(os.environ.get("DEBUG_MODE", "0")) > 0


def setup_logger():
    if not debug_mode_enabled():
        warnings.filterwarnings("ignore")

    logger = logging.getLogger("nebullvm_logger")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [ %(levelname)s ] %(message)s", "%d/%m/%Y %I:%M:%S %p"
    )
    ch.setFormatter(formatter)
    logger.handlers = [ch]
    logger.propagate = False
