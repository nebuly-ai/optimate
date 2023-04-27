import logging
import os
import sys
import warnings
from typing import Any

from loguru import logger


levels_map = {
    0: "ERROR",
    1: "WARNING",
    2: "INFO",
    3: "DEBUG",
}


def debug_mode_enabled():
    return int(os.environ.get("DEBUG_MODE", "0")) > 0


def setup_logger():
    if not debug_mode_enabled():
        warnings.filterwarnings("ignore")

    logging_level = int(os.environ.get("NEBULLVM_LOG_LEVEL", "2"))

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
        level=levels_map[logging_level],
    )
    logger.level("WARNING", color="<fg #d3d3d3>")


class LoggingContext(object):
    def __init__(
        self,
        logger: logging.Logger,
        disabled: bool = False,
        handler: Any = None,
        close: bool = True,
    ):
        self.logger = logger
        self.disabled = disabled
        self.handler = handler
        self.close = close

    def __enter__(self):
        self.logger.disabled = self.disabled
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et: Any, ev: Any, tb: Any):
        if self.disabled is True:
            self.logger.disabled = False
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
