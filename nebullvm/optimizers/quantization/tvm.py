from typing import List, Sequence

from nebullvm.utils.data import DataManager

try:
    from tvm.runtime.ndarray import NDArray
except ImportError:
    NDArray = object


class TVMCalibrator(DataManager):
    def __init__(self, data_reader: Sequence, input_names: List[str]):
        super(TVMCalibrator, self).__init__(data_reader=data_reader)
        self._input_names = input_names

    def __getitem__(self, item: int):
        tuple_ = self._data_reader[item]
        return {name: data for name, data in zip(self._input_names, tuple_)}
