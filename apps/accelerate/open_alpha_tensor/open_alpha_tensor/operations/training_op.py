from pathlib import Path
from typing import Tuple, Any, List

import torch.optim
from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
from open_alpha_tensor.core.training import Trainer
from open_alpha_tensor.operations.checkpoint_op import LoadCheckpointDataOp


class TrainingOperation(Operation):
    """Operation which trains an AlphaTensor model to learn more efficient
    matrix multiplications."""

    def __init__(self):
        super().__init__()
        self._trained_model = None

        self._load_checkpoint_data_op = LoadCheckpointDataOp()

    def execute(
        self,
        model: AlphaTensorModel,
        input_size: int,
        n_steps: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        len_data: int,
        pct_synth: float,
        n_synth_data: int,
        limit_rank: int,
        max_epochs: int,
        n_actors: int,
        mc_n_sim: int,
        N_bar: int,
        last_epoch: int,
        lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        loss_params: Tuple[float, float] = None,
        random_seed: int = None,
        checkpoint_dir: str = None,
        checkpoint_data_dir: str = None,
        n_cob: int = 0,
        cob_prob: float = 0.0,
        data_augmentation: bool = False,
        extra_devices: List[str] = None,
    ):
        """Trains an AlphaTensor model to learn more efficient matrix
        multiplications.

        Args:
            model (AlphaTensorModel): The model to be trained.
            input_size (int): Flattened size of the matrices to be multiplied.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            batch_size (int): Batch size.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (str): The name of the torch device used for training.
            len_data (int): Number of training samples used (both actor
            generated and synthetic).
            pct_synth (float): Initial percentage of synthetic samples used
            for training.
            n_synth_data (int): Number of synthetic training samples.
            limit_rank (int): Maximum rank for synthetically-generated
            matrices.
            max_epochs (int): Number of training epochs.
            n_actors (int): Number of actors to play a single each game at
            each training step.
            mc_n_sim (int): Number of simulations during Monte Carlo tree
            search.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
            last_epoch (int): Latest epoch reached during training from which
            checkpoint data will be loaded.
            lr (float): Learning rate.
            lr_decay_factor (float): Learning rate's decay factor.
            lr_decay_steps (int): Number of learning rate's decay steps.
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function.
            random_seed (int): Randomizing seed.
            checkpoint_dir (str): Directory used to store model checkpoints.
            checkpoint_data_dir (str): Directory used to store games as JSON
            files.
            n_cob (int): Number of change of basis (cob) used for a single
            training sample.
            cob_prob (float): Probability of applying a change of basis.
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation.
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        checkpoint_data_dir = Path(checkpoint_data_dir or "games")
        # build trainer
        trainer = Trainer(
            model=model,
            tensor_size=input_size,
            n_steps=n_steps,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            len_data=len_data,
            pct_synth=pct_synth,
            n_synth_data=n_synth_data,
            limit_rank=limit_rank,
            loss_params=loss_params,
            random_seed=random_seed,
            checkpoint_dir=checkpoint_dir,
            checkpoint_data_dir=checkpoint_data_dir,
            data_augmentation=data_augmentation,
            cob_prob=cob_prob,
            n_cob=n_cob,
            extra_devices=extra_devices,
        )

        # load checkpoint data
        self._load_checkpoint_data_op.execute(
            games_store_dir=checkpoint_data_dir,
            trainer=trainer,
        )

        # train
        trainer.train(
            n_epochs=max_epochs,
            n_games=n_actors,
            mc_n_sim=mc_n_sim,
            N_bar=N_bar,
            starting_epoch=last_epoch,
            initial_lr=lr,
            lr_decay_factor=lr_decay_factor,
            lr_decay_steps=lr_decay_steps,
        )
        self._trained_model = trainer.model

    def get_trained_model(self):
        """Returns the trained model."""
        return self._trained_model

    def get_result(self) -> Any:
        pass
