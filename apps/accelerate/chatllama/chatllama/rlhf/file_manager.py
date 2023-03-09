import os

from beartype.typing import Union, Optional, Tuple

from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
    ConfigTrainer,
)
from chatllama.rlhf.model_list import hf_models

ConfigType = Union[
    Config, ConfigActor, ConfigCritic, ConfigReward, ConfigTrainer
]


class ModelLoader:
    def __init__(
        self,
    ) -> None:
        pass

    def get_model_path(
        self,
        config: ConfigType,
        is_checkpoint: bool = False,
    ) -> Tuple[str, str, str]:
        """Method to get the path to the right model file.
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
        """
        # Get model folder from settings
        model_folder = config.model_folder
        # Create the checkpoint folder if it does not exist
        if is_checkpoint:
            os.path.join(model_folder, "checkpoints")
        # Create the folder for the model type (Actor, Critic, Reward, RLHF)
        if isinstance(config, ConfigReward):
            model_folder = os.path.join(model_folder, "reward")
        elif isinstance(config, ConfigActor):
            model_folder = os.path.join(model_folder, "actor")
        elif isinstance(config, ConfigCritic):
            model_folder = os.path.join(model_folder, "critic")
        elif isinstance(config, ConfigTrainer):
            model_folder = os.path.join(model_folder, "actor_rl")
        else:
            raise ValueError(
                "Model type not recognized during saving or loading"
            )
        if os.path.exists(model_folder) is False:
            os.makedirs(model_folder)
            print(
                f"Model folder does not exist." f"Creating it: {model_folder}"
            )
        # Create the model name
        if config.model in hf_models:
            model_name = os.path.split(config.model)[-1]
        else:
            model_name = config.model
        model_name = f"{model_name}.pt"
        path = os.path.join(model_folder, model_name)
        return model_folder, model_name, path

    def check_model_path(
        self, config: ConfigType, is_checkpoint: bool = False
    ) -> Optional[str]:
        """Method to check if the model path exists,
        called when loadding the model
        """
        model_folder, model_name, path = self.get_model_path(
            config, is_checkpoint
        )
        if os.path.exists(path) is False:
            if is_checkpoint:
                print(
                    f"No previous checkpoint found at"
                    f"{model_folder} for {model_name}"
                )
            else:
                print(
                    f"No previous model found at"
                    f"{model_folder} for model {model_name}"
                )
            return None
        else:
            return path
