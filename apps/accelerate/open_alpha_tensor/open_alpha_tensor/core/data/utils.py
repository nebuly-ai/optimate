from typing import Tuple

import torch


def get_scalars(input_tensor: torch.Tensor, t_step: int, with_bs: bool = True):
    # scalars containing the iteration time
    if with_bs:
        bs = input_tensor.shape[0]
        scalars = torch.zeros((bs, 1))
        scalars[:, 0] = t_step
    else:
        scalars = torch.tensor(t_step).unsqueeze(-1).float()
    return scalars


def map_triplet_to_action(
    triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    base: int,
    n_steps: int,
    add_bias: bool = True,
):
    # map the triplet to an action. First, we concatenate the three tensors and
    # then we convert it to an action using the given base representation. Each
    # element is converted using the formula:
    #   action += element * base^(element_index)
    u, v, w = triplet
    n_dim = u.ndim
    action = torch.cat((u, v, w), dim=-1)
    action = action.reshape(-1, n_steps, action.shape[-1] // n_steps)
    if n_dim == 1:
        action = action.squeeze(0)
    if add_bias:
        action = action + base // 2
    action = action * torch.tensor(
        [base**i for i in range(action.shape[-1])]
    )
    action = action.sum(dim=-1)
    return action


# @torch.jit.script
def _single_action_to_triplet(
    action_val: int,
    basis: int,
    out_dim: int,
    bias: int,
    device: str,
):
    triplet = torch.zeros(out_dim).to(device)
    if action_val > 0:
        idx = int(
            torch.log(torch.tensor(action_val))
            // torch.log(torch.tensor(basis))
        )
    else:
        idx = 0
    while idx >= 0:
        temp = int(basis**idx)
        triplet[idx] = action_val // temp - bias
        action_val = action_val - temp
        idx -= 1
    return triplet


def map_action_to_triplet(
    action_tensor: torch.Tensor,
    cardinality: int = 5,
    vector_size: int = 5,
    add_bias: bool = True,
):
    # map the action to a triplet. The action is converted to a base 5
    # representation and then the three elements are extracted from it.
    # The action has shape (bs, n_steps) and it contains the token for
    # recreating u, v and w. The token is a number between 0 and n_logits.
    action_shape = action_tensor.shape
    action_tensor = action_tensor.reshape(-1)
    if add_bias:
        bias = cardinality // 2
    else:
        bias = 0
    triplets = torch.stack(
        [
            _single_action_to_triplet(
                action_tensor[idx],
                cardinality,
                vector_size,
                bias,
                action_tensor.device,
            )
            for idx in range(len(action_tensor))
        ]
    )
    final_size = triplets.shape[-1]
    return triplets.reshape((*action_shape, final_size))
