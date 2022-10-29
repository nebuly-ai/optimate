import sys
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
