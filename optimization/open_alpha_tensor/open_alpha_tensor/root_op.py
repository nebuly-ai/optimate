from typing import Tuple, List

from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
from open_alpha_tensor.operations.checkpoint_op import LoadCheckPointOp
from open_alpha_tensor.operations.model_op import (
    BuildModelOp,
    SaveModelOp,
    BuildOptimizerOp,
)
from open_alpha_tensor.operations.training_op import TrainingOperation


class TrainAlphaTensorRootOp(Operation):
    """Root operation which trains an AlphaTensor model to learn more
    efficient matrix multiplications."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._optimizer = None

        self._build_model_op = BuildModelOp()
        self._build_optimizer_op = BuildOptimizerOp()
        self._load_checkpoint_op = LoadCheckPointOp()
        self._training_op = TrainingOperation()
        self._save_model_op = SaveModelOp()

    def execute(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
        optimizer_name: str,
        lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        weight_decay: float,
        loss_params: Tuple[float, float],
        checkpoint_dir: str,
        checkpoint_data_dir: str,
        epochs: int,
        batch_size: int,
        len_data: int,
        n_synth_data: int,
        pct_synth: float,
        limit_rank: int,
        n_actors: int,
        mc_n_sim: int,
        N_bar: int,
        device: str,
        save_dir: str,
        random_seed: int,
        n_cob: int,
        cob_prob: float,
        data_augmentation: bool,
        extra_devices: List[str],
    ):
        """Trains an AlphaTensor model to learn more efficient matrix
        multiplications.

        Args:
            tensor_length (int): Number of step tensors fed to the model
            (history and current state),
            input_size (int): Flattened size of the matrices to be multiplied,
            scalars_size (int): Size of the scalar vectors fed to the torso
            model,
            emb_dim (int): Embedding dimension,
            n_steps (int): Number of steps used to get a single action out of
            a triplet,
            n_logits (int): Number of logits output by the policy head,
            n_samples (int): Number of samples used by the policy head at
            evaluation time,
            optimizer_name (str): Name of the optimizer used,
            lr (float): Learning rate,
            lr_decay_factor (float): Learning rate's decay factor,
            lr_decay_steps (int): Number of learning rate's decay steps,
            weight_decay (float): Weight decay used by the optimizer,
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function,
            checkpoint_dir (str): Directory used to store model checkpoints,
            checkpoint_data_dir (str): Directory used to store games as JSON
            files,
            epochs (int): Number of training epochs,
            batch_size (int): Batch size,
            len_data (int): Number of training samples used (both actor
            generated and synthetic),
            n_synth_data (int): Number of synthetic training samples,
            pct_synth (float): Initial percentage of synthetic samples used
            for training,
            limit_rank (int): Maximum rank for synthetically-generated
            matrices,
            n_actors (int): Number of actors to play a single each game at
            each training step,
            mc_n_sim (int): Number of simulations during Monte Carlo tree
            search,
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy,
            device (str): The name of the torch device used for training,
            save_dir (str): Directory where the final trained model will be
            stored,
            random_seed (int): Randomizing seed,
            n_cob (int): Number of change of basis (cob) used for a single
            training sample,
            cob_prob (float): Probability of applying a change of basis,
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation,
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        if self._model is None:
            self._build_model_op.execute(
                tensor_length=tensor_length,
                input_size=input_size,
                scalars_size=scalars_size,
                emb_dim=emb_dim,
                n_steps=n_steps,
                n_logits=n_logits,
                n_samples=n_samples,
            )
            self._model = self._build_model_op.get_model().to(device)

        if self._build_model_op.get_model() is not None:
            self._build_optimizer_op.execute(
                optimizer_name=optimizer_name,
                model=self._build_model_op.get_model(),
                lr=lr,
                weight_decay=weight_decay,
            )
            self._optimizer = self._build_optimizer_op.get_optimizer()

        if self._model is not None and self._optimizer is not None:
            self._load_checkpoint_op.execute(
                self._model, self._optimizer, checkpoint_dir
            )

        if self._load_checkpoint_op.get_model() is not None:
            self._model = self._load_checkpoint_op.get_model()
            self._optimizer = self._load_checkpoint_op.get_optimizer()
            starting_epoch = self._load_checkpoint_op.get_last_epoch()
            self._training_op.execute(
                model=self._model,
                input_size=input_size,
                n_steps=n_steps,
                batch_size=batch_size,
                optimizer=self._optimizer,
                device=device,
                len_data=len_data,
                pct_synth=pct_synth,
                n_synth_data=n_synth_data,
                limit_rank=limit_rank,
                max_epochs=epochs,
                n_actors=n_actors,
                mc_n_sim=mc_n_sim,
                N_bar=N_bar,
                last_epoch=starting_epoch,
                lr=lr,
                lr_decay_factor=lr_decay_factor,
                lr_decay_steps=lr_decay_steps,
                loss_params=loss_params,
                random_seed=random_seed,
                checkpoint_dir=checkpoint_dir,
                checkpoint_data_dir=checkpoint_data_dir,
                n_cob=n_cob,
                cob_prob=cob_prob,
                data_augmentation=data_augmentation,
                extra_devices=extra_devices,
            )
        if self._training_op.get_trained_model() is not None:
            self._model = self._training_op.get_trained_model()
            self._save_model_op.execute(
                model=self._model,
                save_dir=save_dir,
            )

    def get_result(self) -> AlphaTensorModel:
        """Returns the trained torch model"""
        return self._model
