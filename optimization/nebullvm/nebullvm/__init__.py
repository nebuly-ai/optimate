# The torch import is necessary for a strange issue when
# using cuda 11.8, if torch is imported after
# tensorflow it generates a core dumped error
from nebullvm.optional_modules.torch import torch  # noqa F401
from nebullvm.tools.logger import setup_logger

setup_logger()

__all__ = [k for k in globals().keys() if not k.startswith("_")]
