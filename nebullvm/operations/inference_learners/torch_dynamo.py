from pathlib import Path
from typing import Union

from nebullvm.operations.inference_learners.torchscript import (
    TorchScriptInferenceLearner,
)


class TorchDynamoInferenceLearner(TorchScriptInferenceLearner):
    name = "TorchDynamo"

    def save(self, path: Union[str, Path], **kwargs):
        # TODO: Implement save function
        # Saving it like a normal PyTorch model raises this error:
        # https://github.com/pytorch/pytorch/issues/93470
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        # TODO: Implement load function
        raise NotImplementedError
