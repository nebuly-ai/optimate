import logging

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
