import logging
import os
import warnings
from copy import copy

LOGGER_NAME = "nebullvm_logger"


def debug_mode_enabled():
    return int(os.environ.get("DEBUG_MODE", "0")) > 0


def setup_logger():
    level = logging.DEBUG
    if not debug_mode_enabled():
        warnings.filterwarnings("ignore")
        level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(processName)s %(threadName)s [ %(levelname)s ] %(message)s", "%d/%m/%Y %I:%M:%S %p"
    )
    ch.setFormatter(formatter)
    logger.handlers = [ch]
    logger.propagate = False


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def save_root_logger_state():
    orig_root_logger_handlers = copy(logging.getLogger().handlers)
    orig_root_logger_level = copy(logging.getLogger().level)
    return orig_root_logger_handlers, orig_root_logger_level


def raise_logger_level(level=logging.ERROR):
    logging.getLogger().setLevel(level)


def load_root_logger_state(state):
    logging.getLogger().level = state[1]
    logging.getLogger().handlers = state[0]
