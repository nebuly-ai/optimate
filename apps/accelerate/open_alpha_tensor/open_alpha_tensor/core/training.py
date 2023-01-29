from pathlib import Path
from typing import Tuple, List

import torch.optim
import tqdm
from torch.utils.data import DataLoader

from open_alpha_tensor.config import BASE_CHECKPOINT_DATA_DIR, BASE_CHECKPOINT_DIR
from open_alpha_tensor.core.actors.stage import actor_prediction
from open_alpha_tensor.core.data.basis_change import ChangeOfBasis
from open_alpha_tensor.core.data.dataset import TensorGameDataset
from open_alpha_tensor.core.data.generation import f_prob_distribution
from open_alpha_tensor.core.data.utils import map_action_to_triplet
from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel


@torch.no_grad()
def _single_act(
    actor_id: int,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    mc_n_sim: int,
    N_bar: int,
    cob: ChangeOfBasis,
    max_rank: int,
):
    """Executes an episode for a single actor using the MCTS.
    The method is called multiple times in parallel with different actor ids.

    Args:
        actor_id (int): The id of the actor.
        model (torch.nn.Module): The model used to take the action.
        input_tensor (torch.Tensor): State of the game.
        device (str): The name of the torch device used for training.
        mc_n_sim (int): Number of simulations during Monte Carlo tree search.
        N_bar (int): N_bar parameter used to compute tau when improving the
        policy.
        cob (ChangeOfBasis): The change of basis used to generate the input
        tensor.
        max_rank (int): The maximum matrix rank achieved by the actor before
        tree search is stopped.
    """
    print(f"Acting with actor {actor_id}")
    model.to(device)
    cob.device = device
    input_tensor = input_tensor.to(device)
    input_tensor_cob = cob(input_tensor)
    states, policies, rewards = actor_prediction(
        model, input_tensor_cob, max_rank, mc_n_sim, N_bar
    )
    print(f"Actor {actor_id} finished")
    states = [s.to("cpu") for s in states]
    policies = policies.to("cpu")
    rewards = rewards.to("cpu")
    return actor_id, states, policies, rewards


def swap_data(
    states: List[torch.Tensor],
    actions: List[torch.Tensor],
):
    """Swaps the last action with a random one and updates the states
    accordingly for a single game.

    Args:
        states (List[torch.Tensor]): All the states for a single game.
        actions (List[torch.Tensor]): All the actions through the game.
    """
    last_action = actions[-1]
    swap_index = torch.randint(0, len(states) - 1, (1,)).item()
    actions[-1] = actions[swap_index]
    actions[swap_index] = last_action

    actual_state = states[swap_index]
    for i in range(swap_index + 1, len(states) + 1):
        prev_action = actions[i - 1]
        triplet = map_action_to_triplet(
            prev_action, vector_size=actual_state.shape[-1]
        )
        vector_size = actual_state.shape[-1] // 3
        bs = actual_state.shape[0]
        u = triplet[:, :vector_size].reshape(bs, -1, 1, 1)
        v = triplet[:, vector_size : 2 * vector_size].reshape(  # noqa E203
            bs, 1, -1, 1
        )
        w = triplet[:, 2 * vector_size :].reshape(bs, 1, 1, -1)  # noqa E203
        reduced_state = u * v * w
        fut_state = actual_state[:, 0] - reduced_state
        new_state = actual_state[:, 1:].roll(1, dims=1)
        new_state[:, 0] = reduced_state
        actual_state = torch.cat([fut_state, new_state], dim=1)
        states[i] = actual_state
    return states, actions


class Trainer:
    """Trainer for the AlphaTensor model. The trainer does not require an
    explicit loss since the loss is computed by the model itself. The trainer
    is responsible for both the training step and the acting one, storing
    acting performance in a buffer.
    """

    def __init__(
        self,
        model: AlphaTensorModel,
        tensor_size: int,
        n_steps: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        len_data: int,
        pct_synth: float,
        n_synth_data: int,
        limit_rank: int,
        n_cob: int,
        cob_prob: float,
        data_augmentation: bool,
        loss_params: Tuple[float, float] = None,
        random_seed: int = None,
        checkpoint_dir: str = None,
        checkpoint_data_dir: Path = None,
        extra_devices: List[str] = None,
    ):
        """Initializes the trainer.

        Args:
            model (AlphaTensorModel): The model to train.
            tensor_size (int): Flattened size of the matrices to be multiplied.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            batch_size (int): Batch size.
            optimizer (torch.optim.Optimizer): The optimizer used to train the
            model.
            device (str): The name of the torch device used for training.
            len_data (int): Number of training samples used (both actor
            generated and synthetic).
            pct_synth (float): Initial percentage of synthetic samples used
            for training.
            n_synth_data (int): Number of synthetic training samples.
            limit_rank (int): Maximum rank for synthetically-generated
            matrices.
            n_cob (int): Number of change of basis (cob) used for a single
            training sample.
            cob_prob (float): Probability of applying a change of basis.
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation.
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function.
            random_seed (int): Randomizing seed.
            checkpoint_dir (str): Directory used to store model checkpoints.
            checkpoint_data_dir (str): Directory used to store games as JSON
            files.
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataset = TensorGameDataset(
            len_data,
            pct_synth,
            tensor_size,
            n_synth_data,
            limit_rank,
            f_prob_distribution,
            device=device,
            n_steps=n_steps,
            action_memory_len=(model.tensor_length - 1),
            random_seed=random_seed,
        )
        self.batch_size = batch_size
        self.max_rank = limit_rank
        if loss_params is None:
            self.alpha = 1
            self.beta = 1
        else:
            self.alpha, self.beta = loss_params
        self.checkpoint_dir = Path(
            checkpoint_dir if checkpoint_dir else BASE_CHECKPOINT_DIR
        )
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_data_dir = (
            checkpoint_data_dir if checkpoint_data_dir else Path(BASE_CHECKPOINT_DATA_DIR)
        )
        self.checkpoint_data_dir.mkdir(exist_ok=True, parents=True)
        self.change_of_basis = ChangeOfBasis(
            tensor_size, n_cob, cob_prob, device, random_seed
        )
        self.data_augmentation = data_augmentation
        self.extra_devices = extra_devices

    def train_step(self):
        """Executes a single training step by optimizing the current model
        parameters."""
        self.dataset.recompute_synthetic_indexes()
        self.model.train()
        total_loss = 0
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print("Training AlphaTensor")
        for states, scalars, policies, rewards in tqdm.tqdm(dl):
            loss_policy, loss_value = self.model(
                states, scalars, policies, rewards
            )
            loss = self.alpha * loss_policy + self.beta * loss_value
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Total loss: {total_loss}")

    @torch.no_grad()
    def act_step(
        self,
        input_tensor: torch.Tensor,
        n_games: int,
        mc_n_sim: int,
        N_bar: int,
    ):
        """Runs actors in parallel to generate multiple games starting from
        the same input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor used to generate the
            games.
            n_games (int): Number of games to generate / actors to be run in
            parallel.
            mc_n_sim (int): Number of simulations used in the Monte Carlo tree
            search.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
        """
        self.model.eval()
        best_reward = -1e10
        best_game = None

        if self.extra_devices:
            from joblib import Parallel, delayed

            # this means that there is an empty GPU available
            # thus we can use it to parallelize the acting step
            # use joblib to parallelize the acting step
            # we should use _single_act as a function to be parallelized
            extra_devices = (
                self.extra_devices * (n_games // len(self.extra_devices))
                + self.extra_devices[: n_games % len(self.extra_devices)]
            )
            self.model.to("cpu")
            input_tensor = input_tensor.to("cpu")

            print(f"Starting acting phase with {n_games} games")
            results = Parallel(n_jobs=len(self.extra_devices))(
                delayed(_single_act)(
                    actor_id,
                    self.model,
                    input_tensor,
                    extra_devices[actor_id],
                    mc_n_sim,
                    N_bar,
                    self.change_of_basis,
                    self.max_rank,
                )
                for actor_id in range(n_games)
            )
            self.model.to(self.device)

            for actor_id, states, policies, rewards in results:
                if rewards[-1] > best_reward:
                    print(f"New best actor! Actor: {actor_id}")
                    best_reward = rewards[-1]
                    best_game = (states, policies, rewards)
                self.dataset.add_game(states, policies, rewards)
                if self.data_augmentation:
                    states, policies = swap_data(states, policies)
                    self.dataset.add_game(states, policies, rewards)
            if best_game is not None:
                self.dataset.add_best_game(*best_game)
        else:
            for actor_id in range(n_games):
                input_tensor_cob = self.change_of_basis(input_tensor).to(
                    self.device
                )
                print(f"Running actor {actor_id} / {n_games}")
                states, policies, rewards = actor_prediction(
                    self.model,
                    input_tensor_cob,
                    self.max_rank,
                    mc_n_sim,
                    N_bar,
                )
                print(
                    f"Actor {actor_id} finished. Final reward: {rewards[-1]}"
                )
                if rewards[-1] > best_reward:
                    print("New best actor!")
                    best_reward = rewards[-1]
                    best_game = (states, policies, rewards)
                self.dataset.add_game(states, policies, rewards)
                if self.data_augmentation:
                    states, policies = swap_data(states, policies)
                    self.dataset.add_game(states, policies, rewards)
            if best_game is not None:
                self.dataset.add_best_game(*best_game)

    def train(
        self,
        n_epochs: int,
        n_games: int,
        mc_n_sim: int,
        N_bar: int,
        initial_lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        starting_epoch: int = 0,
    ):
        """Trains the model for a given number of epochs.

        Args:
            n_epochs (int): Number of training epochs.
            n_games (int): Number of games to generate / actors to be run in
            parallel at each step.
            mc_n_sim (int): Number of simulations used in the Monte Carlo tree
            search at each step.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
            initial_lr (float): Initial learning rate.
            lr_decay_factor (float): Learning rate's decay factor.
            lr_decay_steps (int): Number of learning rate's decay steps.
            starting_epoch (int, optional): Epoch from which to start / resume
            training.
        """
        self.model = self.model.to(self.device)
        if starting_epoch + 1 > n_epochs // 50:
            self.dataset.change_training_split(0.7, 0.05)
        if (
            starting_epoch + 1 > n_epochs // 10
        ):  # when restarting from a checkpoint
            mc_n_sim = mc_n_sim * 4
        for epoch in range(starting_epoch, n_epochs):
            if epoch + 1 == n_epochs // 50:
                self.dataset.change_training_split(0.7, 0.05)
            if epoch + 1 == n_epochs // 10:
                mc_n_sim = mc_n_sim * 4
            # apply learning rate decay each epoch if epoch < lr_decay_steps
            if 0 < epoch < lr_decay_steps - 1:
                lr = initial_lr * lr_decay_factor ** (epoch / lr_decay_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            print(f"Epoch {epoch} / {n_epochs}")
            self.train_step()
            if epoch % 10 == 0:
                self.act_step(
                    self.dataset.input_tensor, n_games, mc_n_sim, N_bar
                )
            # save checkpoint
            if (epoch + 1) % 100 == 0:
                checkpoint_name = f"checkpoint_{epoch + 1}.pt"
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                torch.save(
                    checkpoint,
                    self.checkpoint_dir / checkpoint_name,
                )
                self.dataset.save_game_data(self.checkpoint_data_dir)
            # exit strategy
            if self.dataset.games_are_good():
                break
        print("Training finished")
