from typing import Callable, List


def map_compilers_and_compressors(ignore_list: List, enum_class: Callable):
    if ignore_list is None:
        ignore_list = []
    else:
        ignore_list = [enum_class(element) for element in ignore_list]
    return ignore_list
