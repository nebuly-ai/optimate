from typing import Sequence, List, Tuple, Any, Union, Iterable

import numpy as np
from loguru import logger

from nebullvm.config import MIN_DIM_INPUT_DATA
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch, Dataset, DataLoader
from nebullvm.tools.onnx import convert_to_numpy


class DataManager:
    """Class for managing the user data in nebullvm.

    Attributes:
        data_reader(Sequence): Object implementing the __getitem__, the
            __len__ and the __iter__/__next__ APIs. It should read the
            user data and return tuples of tensors for feeding the models.
    """

    def __init__(self, data_reader: Sequence):
        self._data_reader = data_reader
        self._pointer = 0
        self.train_idxs = []
        self.test_idxs = []

    def __getitem__(self, item):
        return self._data_reader[item]

    def __len__(self):
        return len(self._data_reader)

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer < len(self):
            data = self[self._pointer]
            self._pointer += 1
            return data
        else:
            raise StopIteration

    def get_numpy_list(
        self, n: int = None, shuffle: bool = False, with_ys: bool = False
    ) -> Union[
        List[Tuple[np.ndarray, ...]], Tuple[List[Tuple[np.ndarray, ...]], List]
    ]:
        if n is None:
            n = len(self)
        if not with_ys:
            return [
                tuple(convert_to_numpy(x) for x in tuple_)
                for tuple_ in self.get_list(n, shuffle)
            ]
        else:
            xs, ys = self.get_list(n, shuffle, with_ys=True)
            return [
                tuple(convert_to_numpy(x) for x in tuple_) for tuple_ in xs
            ], ys

    def get_list(
        self, n: int = None, shuffle: bool = False, with_ys: bool = False
    ) -> Union[List[Tuple[Any, ...]], Tuple[List[Tuple[Any, ...]], List]]:
        if n is None:
            n = len(self)
        if shuffle:
            idx = np.random.choice(len(self), n, replace=n > len(self))
        else:
            idx = np.arange(0, min(n, len(self)))
            if n > len(self):
                np.random.seed(0)
                idx = np.concatenate(
                    [
                        idx,
                        np.random.choice(
                            len(self), n - len(self), replace=True
                        ),
                    ]
                )
        if not with_ys:
            return [self[i][0] for i in idx]

        ys, xs = [], []
        for i in idx:
            x, y = self[i] if len(self[i]) > 1 else (self[i][0], None)
            xs.append(x)
            ys.append(y)
        return xs, ys

    @classmethod
    def from_iterable(cls, iterable: Iterable, max_length: int = 500):
        return cls([x for i, x in enumerate(iterable) if i < max_length])

    @classmethod
    def from_dataloader(
        cls,
        dataloader: Union[DataLoader, tf.data.Dataset],
        max_length: int = 500,
    ):
        batch_size = (
            dataloader.batch_size
            if isinstance(dataloader, DataLoader)
            else dataloader._batch_size
        )

        if batch_size > max_length:
            raise ValueError(
                f"Batch size ({dataloader.batch_size}) is greater than "
                f"max_length ({max_length})."
            )
        data_manager = []
        warning_label = False
        for i, batch in enumerate(dataloader):
            if i * batch_size >= max_length:
                break

            if isinstance(batch, (list, tuple)):
                if len(batch) == 1:
                    data_manager.append((batch, None))
                elif len(batch) == 2:
                    if isinstance(batch[0], tuple):
                        data_manager.append((batch[0], batch[1]))
                    elif isinstance(batch[0], (torch.Tensor, tf.Tensor)):
                        warning_label = True
                        data_manager.append(((batch[0],), batch[1]))
                    else:
                        raise ValueError(
                            "The first element of the batch should be a "
                            "tuple or a torch.Tensor"
                        )
                else:
                    warning_label = True
                    data_manager.append(
                        (tuple(t for t in batch[:-1]), batch[-1])
                    )
            elif isinstance(batch, (torch.Tensor, tf.Tensor)):
                data_manager.append(((batch,), None))
            else:
                raise ValueError(
                    "The batch should be a tuple, a list or a Tensor"
                )

        if warning_label:
            logger.warning(
                "The provided dataloader returns a tuple of tensors"
                "for each batch. The last tensor in the tuple will "
                "be considered as the label. "
                "To avoid this warning, the dataloader should return "
                "a tuple for each batch, where the first element is "
                "a tuple containing the inputs and the second element "
                "is a tensor containing the label."
            )

        return cls(data_manager)

    def get_split(self, split_type="train"):
        return (
            DataManager([self[i] for i in self.train_idxs])
            if split_type == "train"
            else DataManager([self[i] for i in self.test_idxs])
        )

    def split(self, split_pct: float, shuffle: bool = False):
        if shuffle:
            idx = np.random.choice(len(self), len(self), replace=False)
        else:
            idx = np.arange(len(self))

        n = int(round(len(idx) * split_pct))

        if len(self) < MIN_DIM_INPUT_DATA:
            logger.warning(
                f"Not enough data for splitting the DataManager. "
                f"You should provide at least {MIN_DIM_INPUT_DATA} "
                f"data samples to allow a good split between train "
                f"and test sets. Compression, calibration and precision "
                f"checks will use the same data."
            )
            self.train_idxs = idx
            self.test_idxs = idx
        else:
            self.train_idxs = idx[:n]
            self.test_idxs = idx[n:]


class PytorchDataset(Dataset):
    def __init__(self, input_data: DataManager, has_labels: bool = False):
        self.data = input_data
        self.has_labels = has_labels
        self.batch_size = input_data[0][0][0].shape[0]

    def __len__(self):
        return sum([batch_inputs[0].shape[0] for batch_inputs, _ in self.data])

    def __getitem__(self, idx):
        batch_idx = int(idx / self.batch_size)
        item_idx = idx % self.batch_size
        data = tuple([data[item_idx] for data in self.data[batch_idx][0]])

        if self.has_labels:
            label = self.data[batch_idx][1]
            if label is not None:
                return data, self.data[batch_idx][1][item_idx]
            else:
                return data, torch.tensor([0])
        else:
            return data
