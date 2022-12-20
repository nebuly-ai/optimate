from collections import Generator

import torch.utils.data


class ProgressiveTrainingDataset(torch.utils.data.Dataset):
    """Dataset for progressive training."""

    def __init__(self, dataset_generator: Generator):
        with torch.no_grad():
            self.internal_dataset = [
                batch
                for data, sign in dataset_generator
                for batch in zip(data, sign)
            ]

    def __getitem__(self, index):
        return self.internal_dataset[index]

    def __len__(self):
        return len(self.internal_dataset)


def compute_perplexity(tensor: torch.Tensor):
    """Compute perplexity of a tensor. The tensor has shape (batch_size,
    sequence_length, vocab_size).
    The softmax has already been computed over the vocab dimension.
    """
    return torch.exp(-torch.sum(tensor * torch.log(tensor), dim=-1)).mean()
