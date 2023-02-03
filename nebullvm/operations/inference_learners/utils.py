from pathlib import Path
from typing import Union

from nebullvm.operations.inference_learners.base import LearnerMetadata


def load_model(path: Union[Path, str]):
    return LearnerMetadata.read(path).load_model(path)


def save_model(model, path: Union[Path, str]):
    model.save(path)
