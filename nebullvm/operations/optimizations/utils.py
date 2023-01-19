from pathlib import Path
from typing import Callable, List, Union

from nebullvm.operations.inference_learners.base import LearnerMetadata

def map_compilers_and_compressors(ignore_list: List, enum_class: Callable):
    if ignore_list is None:
        ignore_list = []
    else:
        ignore_list = [enum_class(element) for element in ignore_list]
    return ignore_list

def load_model(path: Union[Path, str]):
    return LearnerMetadata.read(path).load_model(path)
