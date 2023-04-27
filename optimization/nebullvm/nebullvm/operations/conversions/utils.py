from nebullvm.core.models import DeepLearningFramework
from nebullvm.operations.conversions.converters import (
    PytorchConverter,
    TensorflowConverter,
    ONNXConverter,
    Converter,
)


def get_conversion_op(framework: DeepLearningFramework) -> Converter:
    if framework == DeepLearningFramework.PYTORCH:
        conversion_op = PytorchConverter()
    elif framework == DeepLearningFramework.TENSORFLOW:
        conversion_op = TensorflowConverter()
    else:
        conversion_op = ONNXConverter()

    return conversion_op
