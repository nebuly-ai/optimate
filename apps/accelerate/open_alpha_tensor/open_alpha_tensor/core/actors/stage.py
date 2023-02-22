from typing import Dict, List

import torch

from open_alpha_tensor.core.data.utils import (
    get_scalars,
    map_action_to_triplet,
)
from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel


def game_is_finished(state):
    """Tells if the game is finished or not.

    Args:
        state (torch.Tensor): The state of the game.
    """
    # state size (1, S, S, S)
    return (state == 0).all()


def remove_duplicates(reducing_tensor: torch.Tensor):
    """Remove duplicates from a tensor.

    Args:
        reducing_tensor (torch.Tensor): The tensor to remove duplicates from.
    """
    # reducing tensor has shape (1, N_mc, S, S, S)
    n_mc = reducing_tensor.shape[1]
    indexes = []
    idx_map = {}
    for idx in range(n_mc):
        if len(indexes) == 0:
            indexes.append(idx)
            idx_map[idx] = []
        else:
            idx_tensor = reducing_tensor[:, idx]
            for index in indexes:
                if (reducing_tensor[:, index] - idx_tensor == 0).all():
                    idx_map[index].append(idx)
                    break
            else:
                indexes.append(idx)
                idx_map[idx] = []

    # idx_map = {i: len(v) for i, v in enumerate(idx_map.values())}
    old_idx_to_new_idx_map = {}
    for new_idx, (key, values) in enumerate(idx_map.items()):
        old_idx_to_new_idx_map[key] = new_idx
        for second_idx in values:
            old_idx_to_new_idx_map[second_idx] = new_idx
    return (
        reducing_tensor[:, indexes],
        old_idx_to_new_idx_map,
        idx_map,
        indexes,
    )


def extract_children_states_from_actions(
    state: torch.Tensor,
    actions: torch.Tensor,
    vec_cardinality: int = 5,
):
    """Extract the children states from the actions.

    Args:
        state (torch.Tensor): The state of the game.
        actions (torch.Tensor): The actions to apply to the state.
        vec_cardinality (int, optional): The cardinality of the vectors.
    """
    # state (1, T, S, S, S)
    # actions (1, K, N_steps)
    # we assume actions to be with N_steps = 1,
    #  and N_logits = |F|^(3S/N_steps). Each action is then mapped in a
    #  unique way to a triplet (u, v, w) where each vector has size S.
    # vector cardinality represents the number of values it can take an entry
    #  of u, v or w.
    bs, k, n_steps = actions.shape[:3]
    len_token = 3 * state.shape[2] // n_steps
    actions = map_action_to_triplet(actions, vec_cardinality, len_token)
    actions = actions.reshape(bs, k, n_steps * len_token)
    vec_dim = state.shape[2]
    u = actions[:, :, :vec_dim].reshape(bs, k, vec_dim, 1, 1)
    v = actions[:, :, vec_dim : 2 * vec_dim].reshape(  # noqa E203
        bs, k, 1, vec_dim, 1
    )
    w = actions[:, :, 2 * vec_dim :].reshape(bs, k, 1, 1, vec_dim)  # noqa E203
    reducing_tensor = u * v * w
    (
        reducing_tensor,
        old_idx_to_new_idx,
        repetition_map,
        not_duplicate_indexes,
    ) = remove_duplicates(reducing_tensor)
    old_state = state[:, 0]
    new_state = old_state.unsqueeze(1) - reducing_tensor
    rolling_states = torch.roll(state, 1)[:, 2:]
    return (
        [
            torch.cat(
                [
                    new_state[:, i : i + 1],  # noqa E203
                    reducing_tensor[:, i : i + 1],  # noqa E203
                    rolling_states,
                ],
                dim=1,
            )
            for i in range(k)
        ],
        old_idx_to_new_idx,
        repetition_map,
        not_duplicate_indexes,
    )


def _reduce_memory_consumption_before_storing(
    possible_states: List[torch.Tensor],
):
    """Reduce the memory consumption before storing the states.

    Args:
        possible_states (List[torch.Tensor]): The possible states.
    """
    final_states = [state[:, 0:2] for state in possible_states]
    previous_actions = possible_states[0][:, 2:]
    storing_dict = {
        "final_states": final_states,
        "previous_actions": previous_actions,
    }
    return storing_dict


def _recompose_possible_states(reduced_memory_states_dict: Dict):
    """Recompose the possible states from the reduced memory states.

    Args:
        reduced_memory_states_dict (Dict): The reduced memory states.
    """
    final_states = reduced_memory_states_dict["final_states"]
    previous_actions = reduced_memory_states_dict["previous_actions"]
    possible_states = [
        torch.cat(
            [
                final_states[i],
                previous_actions,
            ],
            dim=1,
        )
        for i in range(len(final_states))
    ]
    return possible_states


def extract_present_state(state: torch.Tensor) -> torch.Tensor:
    return state[:, 0]


def to_hash(tensor: torch.Tensor) -> str:
    """Converts a tensor to a hash string.

    Args:
        tensor: The tensor to convert.
    """
    hashable_tensor = "_".join(
        tensor.reshape(-1).long().detach().cpu().numpy().astype(str).tolist()
    )
    return hashable_tensor


def from_hash(hashable_tensor: str, shape: tuple) -> torch.Tensor:
    """Converts a hash string back to the original tensor.

    Args:
        hashable_tensor (str): The hash string.
        shape (tuple): The shape of the original tensor.
    """
    return torch.tensor([float(x) for x in hashable_tensor.split("_")]).resize(
        shape
    )


def record_action(tree_dict: Dict, state: str, action: str):
    """Record the action in the tree dictionary.

    Args:
        tree_dict (Dict): The tree dictionary.
        state (str): The state as a hash string.
        action (str): The action as a hash string.
    """
    if state in tree_dict:
        tree_dict[state].append(action)
    else:
        tree_dict[state] = [action]


def select_future_state(
    possible_states: List[torch.Tensor],
    q_values: torch.Tensor,
    N_s_a: torch.Tensor,
    repetitions: Dict[int, list],
    c_1: float = 1.25,
    c_2: float = 19652,
    return_idx: bool = False,
) -> torch.Tensor:
    """Select the future state maximizing the upper confidence bound."""
    # q_values (1, K, 1)
    pi = torch.tensor(
        [
            len(repetitions[i])
            for i in range(len(possible_states))
            if i in repetitions
        ]
    ).to(q_values.device)
    if pi.shape[0] != N_s_a.shape[1]:
        print(pi)
        print(pi.shape, q_values.shape, N_s_a.shape)
        pi = pi[: N_s_a.shape[1]]
    ucb = q_values.reshape(-1) + pi * torch.sqrt(
        torch.sum(N_s_a) / (1 + N_s_a)
    ) * (c_1 + torch.log((torch.sum(N_s_a) + c_2 + 1) / c_2))
    if return_idx:
        return ucb.argmax()
    return possible_states[ucb.argmax()]


@torch.no_grad()
def simulate_game(
    model,
    state: torch.Tensor,
    t_time: int,
    max_steps: int,
    game_tree: Dict,
    states_dict: Dict,
    horizon: int = 5,
):
    """Simulates a game from a given state.

    Args:
        model: The model to use for the simulation.
        state (torch.Tensor): The initial state.
        t_time (int): The current time step.
        max_steps (int): The maximum number of steps to simulate.
        game_tree (Dict): The game tree.
        states_dict (Dict): The states dictionary.
        horizon (int): The horizon to use for the simulation.
    """
    idx = t_time
    max_steps = min(max_steps, t_time + horizon)
    state_hash = to_hash(extract_present_state(state))
    trajectory = []
    # selection
    while state_hash in game_tree:
        (
            possible_states_dict,
            old_idx_to_new_idx,
            repetition_map,
            N_s_a,
            q_values,
            actions,
        ) = states_dict[state_hash]
        possible_states = _recompose_possible_states(possible_states_dict)
        state_idx = select_future_state(
            possible_states, q_values, N_s_a, repetition_map, return_idx=True
        )
        trajectory.append((state_hash, state_idx))  # state_hash, action_idx
        future_state = extract_present_state(possible_states[state_idx])
        state = possible_states[state_idx]
        state_hash = to_hash(future_state)
        idx += 1

    # expansion
    if idx <= max_steps:
        trajectory.append((state_hash, None))
        if not game_is_finished(extract_present_state(state)):
            state = state.to(model.device)
            scalars = get_scalars(state, idx).to(state.device)
            actions, probs, q_values = model(state, scalars)
            (
                possible_states,
                cloned_idx_to_idx,
                repetitions,
                not_dupl_indexes,
            ) = extract_children_states_from_actions(
                state,
                actions,
            )
            not_dupl_actions = actions[:, not_dupl_indexes].to("cpu")
            not_dupl_q_values = torch.zeros(not_dupl_actions.shape[:-1]).to(
                "cpu"
            )
            N_s_a = torch.zeros_like(not_dupl_q_values).to("cpu")
            present_state = extract_present_state(state)
            states_dict[to_hash(present_state)] = (
                _reduce_memory_consumption_before_storing(possible_states),
                cloned_idx_to_idx,
                repetitions,
                N_s_a,
                not_dupl_q_values,
                not_dupl_actions,
            )
            game_tree[to_hash(present_state)] = [
                to_hash(extract_present_state(fut_state))
                for fut_state in possible_states
            ]
            leaf_q_value = q_values
    else:
        leaf_q_value = -int(torch.linalg.matrix_rank(state).sum())
    # backup
    backward_pass(trajectory, states_dict, leaf_q_value=leaf_q_value)


def backward_pass(trajectory, states_dict, leaf_q_value: torch.Tensor):
    """Backward pass of the montecarlo algorithm"""
    reward = 0
    for idx, (state, action_idx) in enumerate(reversed(trajectory)):
        if action_idx is None:  # leaf node
            reward += leaf_q_value
        else:
            (
                _,
                old_idx_to_new_idx,
                _,
                N_s_a,
                q_values,
                _,
            ) = states_dict[state]
            if isinstance(reward, torch.Tensor):
                reward = reward.to(q_values.device)
            action_idx = int(action_idx)
            if action_idx in old_idx_to_new_idx:
                not_dupl_index = old_idx_to_new_idx[int(action_idx)]
            else:
                not_dupl_index = action_idx
            reward -= 1
            q_values[:, not_dupl_index] = (
                N_s_a[:, not_dupl_index] * q_values[:, not_dupl_index] + reward
            ) / (N_s_a[:, not_dupl_index] + 1)
            N_s_a[:, not_dupl_index] += 1


def monte_carlo_tree_search(
    model: torch.nn.Module,
    state: torch.Tensor,
    n_sim: int,
    t_time,
    n_steps: int,
    game_tree: Dict,
    state_dict: Dict,
):
    """Runs the monte carlo tree search algorithm.

    Args:
        model (torch.nn.Module): The model to use for the simulation.
        state (torch.Tensor): The initial state.
        n_sim (int): The number of simulations to run.
        t_time (int): The current time step.
        n_steps (int): The maximum number of steps to simulate.
        game_tree (Dict): The game tree.
        state_dict (Dict): The dictionary containing the states.
    """
    # Note that game tree is not the full tree, but just the one having as root
    #  the current node(state).
    # should we accept also previous updated trajectories for the current node?
    # is it something we should considering when deciding how many simulations
    # we should run? (I think yes)
    state_hash = to_hash(extract_present_state(state))
    if state_hash in state_dict:
        with torch.no_grad():
            N_s_a = state_dict[state_hash][3]
            n_sim -= int(N_s_a.sum())
            n_sim = max(n_sim, 0)

    for _ in range(n_sim):
        simulate_game(model, state, t_time, n_steps, game_tree, state_dict)
    # return next state
    possible_states_dict, _, repetitions, N_s_a, q_values, _ = state_dict[
        state_hash
    ]
    possible_states = _recompose_possible_states(possible_states_dict)
    next_state_idx = select_future_state(
        possible_states, q_values, N_s_a, repetitions, return_idx=True
    )
    next_state = possible_states[next_state_idx]
    return next_state


@torch.no_grad()
def compute_improved_policy(
    state_dict: Dict,
    states: List[str],
    model_n_steps: int,
    model_n_logits: int,
    N_bar: int,
):
    """Compute the improved policy given the state_dict, the list of states.
    The improved policy is computed as (N_s_aˆ(1/tau) / (N_s_aˆ(1/tau)).sum())
    where tau is (log(N_s_a.sum()) / log(N_bar))
    """
    policies = torch.zeros(len(states), model_n_steps, model_n_logits)
    N_bar = torch.tensor(N_bar)
    for idx, state in enumerate(states):
        N_s_a = state_dict[state][3]
        actions = state_dict[state][5]
        if N_s_a.sum() > N_bar:
            tau = (torch.log(N_s_a.sum()) / torch.log(N_bar)).item()
        else:
            tau = 1
        N_s_a = N_s_a ** (1 / tau)
        improved_policy = N_s_a / N_s_a.sum()
        for sample_id in range(actions.shape[1]):
            action_ids = actions[0, sample_id]
            for step_id, action_id in enumerate(action_ids):
                policies[idx, step_id, action_id] += improved_policy[
                    0, sample_id
                ]
    return policies


def actor_prediction(
    model: AlphaTensorModel,
    input_tensor: torch.Tensor,
    maximum_rank: int,
    mc_n_sim: int,
    N_bar: int,
    return_actions: bool = False,
):
    """Runs the monte carlo tree search algorithm to obtain the next states,
    policies and rewards.

    Args:
        model (AlphaTensorModel): The model to use for the simulation.
        input_tensor (torch.Tensor): The initial state.
        maximum_rank (int): The maximum number of steps to simulate.
        mc_n_sim (int): The number of simulations to run.
        N_bar (int): The parameter used to compute the improved policy.
        return_actions (bool): If True, only actions are returned.
    """
    # input_tensor has shape (1, T, S, S, S)
    state = input_tensor
    rank = 0
    game_tree = {}
    state_dict = {}
    hash_states = []
    states = []
    while rank < maximum_rank:
        states.append(state)
        hash_states.append(to_hash(extract_present_state(state)))
        state = monte_carlo_tree_search(
            model,
            state,
            mc_n_sim,
            rank,
            maximum_rank,
            game_tree,
            state_dict,
        )
        if game_is_finished(extract_present_state(state)):
            break
        rank += 1
    final_state = extract_present_state(state)
    policies = compute_improved_policy(
        state_dict, hash_states, model.n_steps, model.n_logits, N_bar
    )
    reward = (
        int(torch.linalg.matrix_rank(final_state).sum())
        if not game_is_finished(final_state)
        else 0
    )
    rewards = torch.cumsum(
        torch.tensor([-1] * (len(policies) - 1) + [reward]), dim=0
    )
    if return_actions:
        actions = [state_dict[hash_state][5] for hash_state in hash_states]
        return actions
    # policies do not have the batch size, but states still have it
    states = [s.squeeze(0) for s in states]
    return states, policies, rewards
