import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from open_alpha_tensor.core.data.generation import generate_synthetic_data
from open_alpha_tensor.core.data.utils import (
    get_scalars,
    map_triplet_to_action,
)

SAVE_DIR_SYNT = str(Path.home() / ".data_alpha_tensor/synthetic_data")


def compute_move(triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Computes the outer product of the three tensors in the triplet that
    will be subtracted from the current state.

    Args:
        triplets (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tensors u,
        v, and w.
    """
    u, v, w = triplets
    return u.reshape(-1, 1, 1) * v.reshape(1, -1, 1) * w.reshape(1, 1, -1)


class SyntheticDataBuffer(Dataset):
    """Dataset of synthetically generated demonstrations."""

    def __init__(
        self,
        tensor_size,
        n_data,
        limit_rank,
        prob_distr,
        n_prev_actions: int,
        device: str,
        n_steps: int,
        random_seed=None,
    ):
        """Builds a dataset of synthetic demonstrations.

        Args:
            tensor_size (int): Size of the tensor.
            n_data (int): Number of demonstrations to generate.
            limit_rank (int): Maximum rank of the generated tensors.
            prob_distr (Callable): Probability distribution to use to generate
            the tensors.
            n_prev_actions (int): Number of previous actions to use as input.
            device (str): Name of the torch device to use.
            n_steps (int): Number of steps to perform in the environment.
            random_seed (int, optional): Random seed to use.
        """
        self.device = device
        self.len_data = 0
        self.n_prev_actions = n_prev_actions
        self.limit_rank = limit_rank
        self.n_steps = n_steps
        self.save_dir = os.path.join(SAVE_DIR_SYNT, f"size_{tensor_size}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        number_of_triplets = len(list(Path(self.save_dir).glob("*.pt"))) // 2
        if number_of_triplets < n_data:
            self.len_data = number_of_triplets
            for i, (output_tensor, list_of_triplets) in enumerate(
                generate_synthetic_data(
                    tensor_size,
                    n_data - number_of_triplets,
                    limit_rank,
                    prob_distr,
                    random_seed,
                )
            ):
                torch.save(
                    output_tensor,
                    os.path.join(
                        self.save_dir, f"output_tensor_{self.len_data}.pt"
                    ),
                )
                torch.save(
                    list_of_triplets,
                    os.path.join(
                        self.save_dir, f"list_of_triplets_{self.len_data}.pt"
                    ),
                )
                self.len_data += 1
        else:
            self.len_data = n_data

    def __len__(self):
        return self.len_data * self.limit_rank

    @torch.no_grad()
    def __getitem__(self, idx):
        i = idx // self.limit_rank
        j = idx % self.limit_rank
        output_tensor = torch.load(
            os.path.join(self.save_dir, f"output_tensor_{i}.pt")
        )
        list_of_triplets = torch.load(
            os.path.join(self.save_dir, f"list_of_triplets_{i}.pt")
        )
        if j != self.limit_rank - 1:
            moves = list_of_triplets[j + 1 :]  # noqa E203
            output_tensor = self._apply_moves(output_tensor, moves)
        triplet = list_of_triplets[j]
        output_tensor = torch.stack(
            [
                output_tensor,
                *(
                    compute_move(t)
                    for t in reversed(
                        list_of_triplets[
                            j + 1 : j + 1 + self.n_prev_actions  # noqa E203
                        ]
                    )
                ),
            ]
        )
        if len(output_tensor) < self.n_prev_actions + 1:
            output_tensor = torch.cat(
                [
                    output_tensor,
                    torch.zeros(
                        self.n_prev_actions + 1 - len(output_tensor),
                        *output_tensor.shape[1:],
                    ),
                ]
            )
        policy = map_triplet_to_action(triplet, base=5, n_steps=self.n_steps)
        reward = torch.tensor([-(j + 1)])
        scalar = get_scalars(output_tensor, self.limit_rank - j, with_bs=False)
        return (
            output_tensor.to(self.device),
            scalar.to(self.device),
            policy.to(self.device),
            reward.to(self.device),
        )

    @staticmethod
    def _apply_moves(
        tensor: torch.Tensor,
        moves: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        """Given an initial state and a list of moves, applies the moves to
        the state.

        Args:
            tensor (torch.Tensor): Initial state.
            moves (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            List of moves.
        """
        for u, v, w in moves:
            tensor = tensor - u.reshape(-1, 1, 1) * v.reshape(
                1, -1, 1
            ) * w.reshape(1, 1, -1)
        return tensor


class GameDataBuffer(Dataset):
    """Buffer to store the data from the games played by the MCTS agent."""

    def __init__(self, device: str, max_buffer_size: int):
        """Initializes the buffer.

        Args:
            device (str): Name of the torch device to use.
            max_buffer_size (int): Maximum size of the buffer.
        """
        self.num_games = 0
        self.temp_dir = tempfile.mkdtemp("game_data_buffer")
        self.game_data = {}
        self.max_buffer_size = max_buffer_size
        self.device = device

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def add_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        """Adds a played game to the buffer.

        Args:
            states (List[torch.Tensor]): Observed game states.
            policies (List[torch.Tensor]): List of policies.
            rewards (List[torch.Tensor]): Observed rewards.
        """
        self.game_data[self.num_games] = len(states)
        torch.save(
            states, os.path.join(self.temp_dir, f"states_{self.num_games}.pt")
        )
        torch.save(
            policies,
            os.path.join(self.temp_dir, f"policies_{self.num_games}.pt"),
        )
        torch.save(
            rewards,
            os.path.join(self.temp_dir, f"rewards_{self.num_games}.pt"),
        )
        self.num_games += 1
        if self.num_games >= self.max_buffer_size:
            # remove oldest game. Note that this line is not thread safe. Lock
            # should be added if multiple threads are used.
            self.num_games = 0

    def __len__(self):
        return sum(self.game_data.values())

    @torch.no_grad()
    def __getitem__(self, idx):
        i = 0
        while idx >= self.game_data[i]:
            idx -= self.game_data[i]
            i += 1
        states = torch.load(os.path.join(self.temp_dir, f"states_{i}.pt"))
        policies = torch.load(os.path.join(self.temp_dir, f"policies_{i}.pt"))
        rewards = torch.load(os.path.join(self.temp_dir, f"rewards_{i}.pt"))
        return (
            states[idx].to(self.device),
            get_scalars(states[idx], idx, with_bs=False).to(self.device),
            policies[idx].to(self.device).argmax(dim=-1),
            rewards[idx].to(self.device).reshape(1),
        )

    def save_game_data(self, path: str):
        """Copy save_dir content in path and save game_data
        in json format
        """
        shutil.copytree(self.temp_dir, path, dirs_exist_ok=True)
        with open(os.path.join(path, "game_data.json"), "w") as f:
            json.dump(self.game_data, f)

    def load_game_data(self, path: str):
        """Load game_data from json format and copy content
        in save_dir
        """
        with open(os.path.join(path, "game_data.json"), "r") as f:
            self.game_data = json.load(f)
        shutil.copytree(path, self.temp_dir)
        self.num_games = len(self.game_data)


class TensorGameDataset(Dataset):
    """Dataset to be used for training the AlphaTensor algorithm using both
    actor generated and synthetic data. A basis change can be applied to both
    the data type with a probability specified in the constructor. The
    synthetic data and the actor generated one are stored in two data buffers.
    """

    def __init__(
        self,
        len_data,
        pct_synth,
        tensor_size,
        n_synth_data,
        limit_rank,
        prob_distr,
        action_memory_len: int,
        device: str,
        n_steps: int,
        random_seed=None,
    ):
        self.synthetic_data_buffer = SyntheticDataBuffer(
            tensor_size,
            n_synth_data,
            limit_rank,
            prob_distr,
            action_memory_len,
            n_steps=n_steps,
            device=device,
            random_seed=random_seed,
        )
        self.game_data_buffer = GameDataBuffer(
            device=device, max_buffer_size=100000
        )
        self.best_game_data_buffer = GameDataBuffer(
            device=device, max_buffer_size=1000
        )
        self.len_data = len_data
        self.pct_synth = pct_synth
        self.pct_best_game = 0
        self.synth_bool = torch.ones(len_data, dtype=torch.bool)
        self.synth_idx = torch.from_numpy(
            np.random.choice(
                len(self.synthetic_data_buffer), len_data, replace=False
            )
        )
        self.game_idx = None
        self.best_game_idx = None
        self.action_memory_len = action_memory_len
        self.tensor_size = tensor_size
        self.device = device

    def change_training_split(self, pct_synth, pct_best_game):
        self.pct_synth = pct_synth
        self.pct_best_game = pct_best_game

    def recompute_synthetic_indexes(self):
        if len(self.game_data_buffer) > 0:
            self.synth_bool = torch.rand(self.len_data) < self.pct_synth
            len_synth_data = self.synth_bool.sum().item()
            self.synth_idx = torch.from_numpy(
                np.random.choice(
                    len(self.synthetic_data_buffer),
                    len_synth_data,
                    replace=False,
                )
            )
            if len(self.best_game_data_buffer) > 0 and self.pct_best_game > 0:
                len_game_data = int(
                    (1 - self.pct_synth - self.pct_best_game) * self.len_data
                )
                replace_game = len_game_data > len(self.game_data_buffer)
                len_best_game_data = (
                    self.len_data - len_synth_data - len_game_data
                )
                replace_best_game = len_best_game_data > len(
                    self.best_game_data_buffer
                )
                self.game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.game_data_buffer),
                        len_game_data,
                        replace=replace_game,
                    )
                )
                self.best_game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.best_game_data_buffer),
                        len_best_game_data,
                        replace=replace_best_game,
                    )
                )
            else:
                len_game_data = self.len_data - len_synth_data
                replace_game = len_game_data > len(self.game_data_buffer)
                self.game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.game_data_buffer),
                        len_game_data,
                        replace=replace_game,
                    )
                )

    def __getitem__(self, idx):
        if self.synth_bool[idx]:
            return self.synthetic_data_buffer[
                self.synth_idx[self.synth_bool[:idx].sum()]
            ]
        else:
            if self.pct_best_game > 0 and self.best_game_idx is not None:
                if idx - self.synth_bool[:idx].sum() < len(self.best_game_idx):
                    return self.best_game_data_buffer[
                        self.best_game_idx[idx - self.synth_bool[:idx].sum()]
                    ]
                else:
                    return self.game_data_buffer[
                        self.game_idx[
                            idx
                            - self.synth_bool[:idx].sum()
                            - len(self.best_game_idx)
                        ]
                    ]
            else:
                return self.game_data_buffer[
                    self.game_idx[idx - self.synth_bool[:idx].sum()]
                ]

    def __len__(self):
        return self.len_data

    def add_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        self.game_data_buffer.add_game(states, policies, rewards)

    def add_best_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        self.best_game_data_buffer.add_game(states, policies, rewards)

    def save_game_data(self, path):
        self.game_data_buffer.save_game_data(os.path.join(path, "game_data"))
        self.best_game_data_buffer.save_game_data(
            os.path.join(path, "best_game_data")
        )

    def load_game_data(self, path):
        self.game_data_buffer.load_game_data(os.path.join(path, "game_data"))
        self.best_game_data_buffer.load_game_data(
            os.path.join(path, "best_game_data")
        )

    @property
    def input_tensor(self) -> torch.Tensor:
        max_matrix_size = int(np.sqrt(self.tensor_size))
        input_tensor = torch.zeros(
            1,
            self.action_memory_len + 1,
            self.tensor_size,
            self.tensor_size,
            self.tensor_size,
        )
        matrix_dims = (
            torch.randint(1, max_matrix_size, (3,))
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        operation_tensor = self._build_tensor_game_input(
            *matrix_dims, action_memory_len=self.action_memory_len
        )

        input_tensor[
            0,
            :,
            : operation_tensor.shape[1],
            : operation_tensor.shape[2],
            : operation_tensor.shape[3],
        ] = operation_tensor
        return input_tensor.to(self.device)

    @staticmethod
    def _build_tensor_game_input(
        dim_1: int, dim_k: int, dim_2: int, action_memory_len: int
    ):
        """Build the input tensor for the game. The input tensor has shape
        (action_memory_len+1, matrix_size**2, matrix_size**2, matrix_size**2).
        The first slice represent the matrix multiplication tensor which will
        be reduced by the TensorGame algorithm. The other slices represent the
        action memory.
        """
        input_tensor = torch.zeros(
            action_memory_len + 1, dim_1 * dim_k, dim_k * dim_2, dim_1 * dim_2
        )
        for r in range(dim_1 * dim_2):
            for k in range(dim_k):
                input_tensor[
                    0, (r // dim_2) * dim_k + k, k * dim_2 + r % dim_2, r
                ] = 1
        return input_tensor

    def games_are_good(self):
        return False
