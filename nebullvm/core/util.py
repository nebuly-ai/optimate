import json
from contextlib import contextmanager
from typing import Any


@contextmanager
def patched_attr(obj: object, attr_name: str, new_attr: any):
    old_method = getattr(obj, attr_name)
    setattr(obj, attr_name, new_attr)
    yield
    setattr(obj, attr_name, old_method)


def is_json_serializable(obj: Any) -> bool:
    primitive_types = (int, float, str, bool, type(None))
    if type(obj) in primitive_types:
        return True
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False
