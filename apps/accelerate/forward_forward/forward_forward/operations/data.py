import urllib.request
from typing import Any

import torch
import torch.utils.data
from nebullvm.operations.base import Operation
from torchvision import datasets, transforms


class MNISTDataLoaderOperation(Operation):
    """DataLoaderOperation"""

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None

    def get_result(self) -> Any:
        if self.train_data is not None:
            return self.train_data, self.test_data
        else:
            return None

    def execute(self, batch_size: int, shuffle: bool):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=1000,
            shuffle=False,
        )
        self.train_data = train_loader
        self.test_data = test_loader


def download_fables():
    http_str = "http://classics.mit.edu/Aesop/fab.mb.txt"
    with urllib.request.urlopen(http_str) as response:
        html = response.read()
    return html.decode("utf-8")


def get_fables():
    fables = download_fables()
    fables = fables.split("SECTION 1")[1]
    fables = fables.split("THE END")[0]
    fables = fables.split("\n\n")
    fables = [fable for fable in fables if len(fable) >= 100]
    return fables


VOCABULARY = {
    " ": 0,
    "!": 1,
    ",": 2,
    ".": 3,
    "a": 4,
    "b": 5,
    "c": 6,
    "d": 7,
    "e": 8,
    "f": 9,
    "g": 10,
    "h": 11,
    "i": 12,
    "j": 13,
    "k": 14,
    "l": 15,
    "m": 16,
    "n": 17,
    "o": 18,
    "p": 19,
    "q": 20,
    "r": 21,
    "s": 22,
    "t": 23,
    "u": 24,
    "v": 25,
    "w": 26,
    "x": 27,
    "y": 28,
    "z": 29,
}


def tokenize(fable, max_len=100):
    tokenized_fable = [
        VOCABULARY[char]
        for i, char in enumerate(fable.lower())
        if char in VOCABULARY
    ]
    return tokenized_fable[:max_len]


def get_tokenized_fables():
    fables = get_fables()
    tokenized_fables = [tokenize(fable) for fable in fables]
    tokenized_fables = torch.stack(
        [
            torch.tensor(tokens)
            for tokens in tokenized_fables
            if len(tokens) == 100
        ]
    )
    return tokenized_fables


def get_dataloader(batch_size=32, test_size=0.2, shuffle=True):
    tokenized_fables = get_tokenized_fables()
    n_test = int(len(tokenized_fables) * test_size)
    test_set = torch.utils.data.TensorDataset(tokenized_fables[:n_test])
    train_set = torch.utils.data.TensorDataset(tokenized_fables[n_test:])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=n_test, shuffle=False
    )
    return train_loader, test_loader


class AesopFablesDataLoaderOperation(Operation):
    """DataLoaderOperation"""

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None

    def get_result(self) -> Any:
        if self.train_data is not None:
            return self.train_data, self.test_data
        else:
            return None

    def execute(self, batch_size: int, shuffle: bool):
        train_loader, test_loader = get_dataloader(
            batch_size=batch_size, test_size=0.2, shuffle=shuffle
        )
        self.train_data = train_loader
        self.test_data = test_loader
