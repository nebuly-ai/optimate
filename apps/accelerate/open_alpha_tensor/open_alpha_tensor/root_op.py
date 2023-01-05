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
            self._model = self._build_model_op.get_model()

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
                n_cob=n_cob,
                cob_prob=cob_prob,
                data_augmentation=data_augmentation,
                extra_devices=extra_devices,
            )

    def get_result(self) -> AlphaTensorModel:
        return self._model
