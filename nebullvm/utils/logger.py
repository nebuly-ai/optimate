import logging
import os
import warnings
from copy import copy


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


def save_root_logger_state():
    orig_root_logger_handlers = copy(logging.getLogger().handlers)
    orig_root_logger_level = copy(logging.getLogger().level)
    return orig_root_logger_handlers, orig_root_logger_level


def raise_logger_level(level=logging.ERROR):
    logging.getLogger().setLevel(level)


def load_root_logger_state(state):
    logging.getLogger().level = state[1]
    logging.getLogger().handlers = state[0]
