from pathlib import Path
from typing import Callable

import numpy as np
import torch


def get_change_basis_matrix(
    tensor_size: int,
    n_cob: int,
    entry_distribution: Callable = torch.randn,
    random_seed: int = None,
):
    """Generate a list of change of basis matrices.

    Args:
        tensor_size (int): Size of the tensor.
        n_cob (int): Number of change of basis matrices.
        entry_distribution (Callable, optional): Distribution of the entries
        of the change of basis matrices.
        random_seed (int, optional): Random seed for reproducibility.
    """
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    for _ in range(n_cob):
        diag_p = 2 * (torch.rand(tensor_size) > 0.5).float() - 1
        diag_l = 2 * (torch.rand(tensor_size) > 0.5).float() - 1
        random_matrix = entry_distribution((tensor_size, tensor_size))
        p_matrix = torch.diag(diag_p)
        l_matrix = torch.diag(diag_l)
        p_matrix = p_matrix + torch.triu(random_matrix, diagonal=1)
        l_matrix = l_matrix + torch.tril(random_matrix, diagonal=-1)
        yield torch.matmul(p_matrix, l_matrix)


def cob_entry_prob_distribution(size):
    full_size = int(np.prod(size))
    vals = torch.tensor([-1, 0, 1])
    probs = torch.tensor([0.0075, 0.985, 0.0075]).unsqueeze(0)
    cum_sum = torch.cumsum(probs, dim=-1)
    unif_prob = torch.rand((full_size, 1))
    tensor_idx = torch.argmax((unif_prob <= cum_sum).int(), dim=1)
    tensor = vals[tensor_idx]
    return tensor.reshape(size)


class ChangeOfBasis:
    """Change of Basis class."""

    """Change of Basis class."""

    def __init__(
        self,
        tensor_size: int,
        n_cob: int,
        cob_prob: float,
        device: str,
        random_seed: int = None,
    ):
        """Builds a ChangeOfBasis object.

        Args:
            tensor_size (int): Size of the tensor.
            n_cob (int): Number of change of basis matrices.
            cob_prob (float): Probability of applying a change of basis.
            device (str): Name of the torch device to use.
            random_seed (int, optional): Random seed for reproducibility.
        """
        self.tmp_dir = Path.home() / ".data_alpha_tensor/cob_matrices"
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        for i, cob_matrix in enumerate(
            get_change_basis_matrix(
                tensor_size, n_cob, cob_entry_prob_distribution, random_seed
            )
        ):
            torch.save(cob_matrix, f"{self.tmp_dir}/cob_matrix_{i}.pt")
        self.tensor_size = tensor_size
        self.n_cob = n_cob
        self.cob_prob = cob_prob
        self.device = device

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, return_basis: bool = False):
        """Apply a change of basis to a tensor.

        Args:
            tensor (torch.Tensor): Tensor to apply the change of basis to.
            return_basis (bool, optional): Whether to return the change of
            basis matrix as well.
        """
        cob_prob = torch.rand(1).item()
        if cob_prob > self.cob_prob:
            return tensor
        random_cob = torch.randint(low=0, high=self.n_cob, size=(1,))
        cob_matrix = torch.load(
            f"{self.tmp_dir}/cob_matrix_{int(random_cob)}.pt"
        ).to(self.device)

        # apply change of basis to each tensor dimension
        inner_tensor = tensor[0, 0]
        tensor_size = inner_tensor.shape[-1]
        original_shape = inner_tensor.shape
        cob_matrix = cob_matrix.transpose(0, 1)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(0, 2, 1)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(2, 1, 0)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(2, 0, 1)
        tensor[0, 0] = inner_tensor
        if return_basis:
            return tensor, cob_matrix.transpose(0, 1)
        return tensor
