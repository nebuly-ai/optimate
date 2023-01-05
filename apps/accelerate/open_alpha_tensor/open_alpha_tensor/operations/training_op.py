from pathlib import Path
from typing import Tuple, Any, List

import torch.optim
from nebullvm.operations.base import Operation

from open_alpha_tensor.core.modules.alpha_tensor import AlphaTensorModel
from open_alpha_tensor.core.training import Trainer
from open_alpha_tensor.operations.checkpoint_op import LoadCheckpointDataOp


class TrainingOperation(Operation):
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
        return self._trained_model

    def get_result(self) -> Any:
        pass
