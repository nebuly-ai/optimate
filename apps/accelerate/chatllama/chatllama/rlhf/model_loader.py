import os
import shutil

from beartype.typing import Union, Optional, Tuple

from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)
from chatllama.rlhf.model_list import hf_models

ConfigType = Union[Config, ConfigActor, ConfigCritic, ConfigReward]


class ModelLoader:
    """Class to load and save models and their checkpoints during training."""

    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def get_training_stats_path(config: ConfigType) -> str:
        """Method to get the path to the training stats file. Used when saving

        Args:
            config (ConfigType): the config object
        """
        model_folder, model_name, path = ModelLoader.get_model_path(
            config, is_checkpoint=True
        )
        stat_path = os.path.join(model_folder, "training_stats.json")
        return stat_path

    @staticmethod
    def look_for_last_checkpoint(
        model_folder: str,
        model_name: str,
    ) -> Optional[str]:
        """Method to look for the last checkpoint in the model folder
        checkpoint are saved as {model_name}_epoch_{current_epoch}.pt

        Args:
            model_folder (str): the folder where the checkpoints are saved
            model_name (str): the name of the model
        """
        # remove .pt to model name
        model_name = model_name.split(".")[0]
        checkpoints = [
            f for f in os.listdir(model_folder) if f.startswith(model_name)
        ]
        if len(checkpoints) == 0:
            return None
        else:
            checkpoints = sorted(checkpoints)
            # get last checkpoint
            last_checkpoint = checkpoints[-1]
            return last_checkpoint

    @staticmethod
    def look_for_checkpoint_by_name(
        model_folder: str,
        checkpoint_name: str,
    ) -> Optional[str]:
        """Method to look for a particular checkpoint in the model folder
        checkpoint are saved as
        {model_name}_epoch_{current_epoch}_steps_{current_steps}.pt

        Args:
            model_folder (str): the folder where the checkpoints are saved
            checkpoint_name (str): the name of the checkpoint
        """
        # look for a file named checkpoint_name in the model folder
        path = os.path.join(model_folder, checkpoint_name)
        if os.path.exists(path):
            return checkpoint_name
        else:
            return None

    @staticmethod
    def get_checkpoint_name(config: ConfigType) -> str:
        if isinstance(config, Config):
            return config.trainer.checkpoint_name
        else:
            return config.checkpoint_name

    @staticmethod
    def get_base_model_folder_from_config(config: ConfigType) -> str:
        if isinstance(config, ConfigActor) or isinstance(config, ConfigReward):
            return config.model_folder
        elif isinstance(config, Config):
            return config.actor.model_folder
        else:
            raise ValueError(
                "Config type not recognized during saving or loading"
            )

    @staticmethod
    def get_model_type_from_config(config: ConfigType) -> str:
        if isinstance(config, ConfigReward):
            # here use ad-hoc flag from config to distinguish between
            #  reward and critic
            if config.is_reward:
                return "reward"
            else:
                return "critic"
        elif isinstance(config, ConfigActor):
            return "actor"
        elif isinstance(config, Config):
            return "actor_rl"

    @staticmethod
    def get_model_name_from_config(config: ConfigType) -> str:
        model_name = None
        if isinstance(config, Config):
            model_name = config.actor.model
        elif isinstance(config, ConfigReward) or isinstance(
            config, ConfigActor
        ):
            model_name = config.model
        if model_name in hf_models:
            return os.path.split(model_name)[-1]
        if model_name is None:
            raise ValueError("Model name not found")
        return model_name

    @staticmethod
    def delete_old_checkpoints(
        model_folder: str, model_name: str, n_ckp_to_keep: int = 5
    ):
        """Method to discard old checkpoints, keeping only the last
        n_ckp_to_keep

        Args:
            model_folder (str): the folder where the checkpoints are saved
            model_name (str): the name of the model
            n_ckp_to_keep (int): the number of checkpoints to keep
        """

        # remove .pt to model name
        model_name = model_name.split(".")[0]
        checkpoints = [
            f for f in os.listdir(model_folder) if f.startswith(model_name)
        ]
        if len(checkpoints) == 0:
            return
        else:
            checkpoints = sorted(checkpoints)
            # check if the number of checkpoint is greater than 5
            if len(checkpoints) > n_ckp_to_keep:
                for c in checkpoints[:-n_ckp_to_keep]:
                    checkpoint_path = os.path.join(model_folder, c)
                    os.remove(checkpoint_path)

    @staticmethod
    def get_model_path(
        config: ConfigType,
        is_checkpoint: bool = False,
        current_epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        max_epochs: int = 1_000_000_000,
        max_steps: int = 1_000_000_000,
    ) -> Tuple[str, str, Optional[str]]:
        """Method to get the path to the right model file. Used when saving
        the model.
        The hierarchy of the model folder is:
        -- model_folder: here store the models trained, for each type of model
                        there is a dedicated folder
            -- actor
            -- critic
            -- reward
            -- actor_rl
            -- checkpoints: here store the checkpoints during training, for
                            each type of model there is a dedicated folder
                -- actor
                -- critic
                -- reward
                -- actor_rl

        Args:
            config (ConfigType): the config object, contains info of the model
            is_checkpoint (bool): if True, the path is for a checkpoint
            current_epoch (Optional[int]): the current epoch, used to create
                the checkpoint name. If is_checkpoint is True, and
                current_epoch is None, return just the folder and the simple
                model name for the possible checkpoint.
            current_step (Optional[int]): the current step, used to create
                the checkpoint name.
            max_epochs (Optional[int]): the maximum number of epochs, used to
                create the checkpoint name.
            max_steps (Optional[int]): the maximum number of steps, used to
                create the checkpoint name.

        Returns:
            model_folder (str): the folder where the model is saved
            model_name (str): the name of the model
            path (Optional[str]): the path to the model. If is_checkpoint is
                True, and current_epoch is None, return None
        """
        model_folder = ModelLoader.get_base_model_folder_from_config(config)

        # Add the checkpoint path if necessary
        if is_checkpoint:
            model_folder = os.path.join(model_folder, "checkpoints")

        # Create the folder for the model type
        #  (Actor, Critic, Reward, Actor_RL)
        model_type = ModelLoader.get_model_type_from_config(config)
        model_folder = os.path.join(model_folder, model_type)

        # Make the path if not exists
        if os.path.exists(model_folder) is False:
            os.makedirs(model_folder)
            print(f"Model folder does not exist. Creating it: {model_folder}")

        # Create the model name
        model_name = ModelLoader.get_model_name_from_config(config)

        # If is a checkpoint and current epoch are available
        # extend the model name with the epoch, if none epoch is provided
        # just return the simple model name
        if is_checkpoint and current_epoch is not None:
            # number of characters to store the checkpoints
            n_char = max(len(str(max_epochs)), len(str(max_steps)))
            # create the string epoch such that it is always the same length
            # equalt to n_char (i.e. 00000001) necessary for sorting
            string_epoch = str(current_epoch)
            string_epoch = "0" * (n_char - len(string_epoch)) + string_epoch
            string_epoch = f"_epoch_{string_epoch}"
            if current_step is not None:
                string_step = str(current_step)
                string_step = "0" * (n_char - len(string_step)) + string_step
                string_step = f"_step_{string_step}"
                model_name = f"{model_name}{string_epoch}{string_step}.pt"
            else:
                model_name = f"{model_name}{string_epoch}.pt"
        else:
            model_name = f"{model_name}.pt"

        # if the epoch is not provided, and it is a checkpoint
        # is impossible to know the path to the file.
        # but we can know the model folder and the model name
        if is_checkpoint and current_epoch is None:
            path = None
        else:
            path = os.path.join(model_folder, model_name)
        return model_folder, model_name, path

    @staticmethod
    def check_model_path(
        config: ConfigType,
        is_checkpoint: bool = False,
        current_epoch: Optional[int] = None,
        current_step: Optional[int] = None,
    ) -> Optional[int]:
        """Method to check if the model path exists to load models
        or checkpoints.

        Args:
            config (ConfigType): the config object, contains info of the model
            is_checkpoint (bool): if True, the path is for a checkpoint
            current_epoch (Optional[int]): the current epoch.
                is is_checkpoint is True, and current_epoch is None,
                it will look for the last checkpoint and return it.

        Returns:
            path (Optional[str]): the path to the model. If is_checkpoint is
                True, and current_epoch is None, search for the last checkpoint
                and return it. If no checkpoint is found, return None.
            epoch (Optional[int]): the epoch of the checkpoint if an actual
                checkpoint is found. If no checkpoint is found, return None.
        """
        model_folder, model_name, path = ModelLoader.get_model_path(
            config,
            is_checkpoint,
            current_epoch,
        )

        # If i am looking for a checkpoint.
        if is_checkpoint and current_epoch is None:
            # If the checkpoint is specified by name use it
            checkpoint_name = ModelLoader.get_checkpoint_name(config)
            if checkpoint_name is not None:
                checkpoint = ModelLoader.look_for_checkpoint_by_name(
                    model_folder, checkpoint_name
                )
            else:
                checkpoint = ModelLoader.look_for_last_checkpoint(
                    model_folder, model_name
                )
            if checkpoint is not None:
                path = os.path.join(model_folder, checkpoint)
                # Get the epoch number from the checkpoint name

        if path is not None:
            if os.path.exists(path) is False:
                path = None

        if path is None:
            if is_checkpoint:
                checkpoint_name = ModelLoader.get_checkpoint_name(config)
                if checkpoint_name is not None:
                    print(
                        f"No checkpoint found at {model_folder} "
                        f"with name {config.checkpoint_name}"
                    )
                else:
                    print(
                        f"No previous checkpoint found at "
                        f"{model_folder} for {model_name}"
                    )
            else:
                print(
                    f"No previous model found at "
                    f"{model_folder} for model {model_name}"
                )
        else:
            if is_checkpoint:
                # the name is modelname_epoch_00000001_step_00000001.pt
                # or modelname_epoch_00000001.pt
                if "_step_" in path:
                    epoch = int(path.split("_epoch_")[-1].split("_")[0])
                    step = int(path.split("_step_")[-1].split(".")[0])
                    print(
                        f"Found checkpoint for epoch {epoch + 1},"
                        f" step {step + 1}..."
                    )
                else:
                    epoch = int(path.split("_epoch_")[-1].split(".")[0])
                    print(f"Found checkpoint for epoch {epoch + 1} ...")
            else:
                print(f"Found model at {path}")
        return path

    def init_critic_from_reward(config: ConfigCritic) -> None:
        """Method to initialize the critic from the reward model.
        If the critic folder is empty
        """

        if config.is_reward is True:
            raise ValueError(
                "The config should work for the Critic model,"
                "but the config seems to be for the Reward model"
            )

        # check that the critic folder is empty
        path = ModelLoader.check_model_path(config)
        _, _, critic_path = ModelLoader.get_model_path(config)
        if path is None:
            print("Initializing Critic from Reward model...")
            config.is_reward = True
            path = ModelLoader.check_model_path(config)
            if path is not None:
                _, _, reward_path = ModelLoader.get_model_path(config)
                # copy the file in reward_path to critic_path
                shutil.copy(reward_path, critic_path)
            else:
                print("Critic Model remains uninitialized")
        config.is_reward = False
