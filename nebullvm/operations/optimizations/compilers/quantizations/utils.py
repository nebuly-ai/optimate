from loguru import logger

from nebullvm.core.models import QuantizationType


def check_quantization(
    quantization_type: QuantizationType, perf_loss_ths: float
):
    if quantization_type is not None and perf_loss_ths is None:
        logger.warning(
            "Got a valid quantization type without any given quantization "
            "threshold. The quantization step will be ignored."
        )
