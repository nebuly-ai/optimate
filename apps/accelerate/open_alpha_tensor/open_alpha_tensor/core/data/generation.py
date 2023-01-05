from typing import Callable

import torch


def generate_synthetic_data(
    tensor_size: int,
    n_data: int,
    limit_rank: int,
    prob_distr: Callable = torch.randn,
    random_seed: int = None,
):
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    for _ in range(n_data):
        # rank = torch.randint(low=1, high=limit_rank + 1, size=(1,)).item()
        rank = limit_rank
        output_tensor = torch.zeros(tensor_size, tensor_size, tensor_size)
        list_of_triplets = []
        for i in range(rank):
            valid_triplet = False
            while not valid_triplet:
                u = prob_distr(tensor_size)
                v = prob_distr(tensor_size)
                w = prob_distr(tensor_size)
                generated_tensor = (
                    u.reshape(-1, 1, 1)
                    * v.reshape(1, -1, 1)
                    * w.reshape(1, 1, -1)
                )
                if not (generated_tensor == 0).all():
                    valid_triplet = True
                    list_of_triplets.append((u, v, w))
                    output_tensor += generated_tensor
        yield output_tensor, list_of_triplets


def f_prob_distribution(size):
    f_vals = torch.tensor([-2, -1, 0, 1, 2])
    f_probs = torch.tensor([0.001, 0.099, 0.8, 0.099, 0.001]).unsqueeze(0)
    f_cum_sum = torch.cumsum(f_probs, dim=-1)
    unif_prob = torch.rand((size, 1))
    tensor_idx = torch.argmax((unif_prob <= f_cum_sum).int(), dim=1)
    tensor = f_vals[tensor_idx]
    return tensor


def z2_prob_distribution(size):
    return (torch.rand(size) > 0.5).int()
