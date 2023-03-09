import os

from beartype.typing import Union, Optional

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

    def get_model_name(
        self,
        model_string: str,
    ) -> str:
        # if the model is hf model, remove the "/"
        # otherwise return just the name
        if model_string in hf_models:
            model_name = os.path.split(model_string)[-1]
        else:
            model_name = model_string
        # add extension
        return f"{model_name}.pt"

    def get_model_folder(
        self,
        model_folder: str,
        is_checkpoint: bool = False,
    ) -> str:
        # Create the model folder if it does not exist
        if os.path.exists(model_folder) is False:
            os.makedirs(model_folder)
            print(
                f"Model folder does not exist." f"Creating it: {model_folder}"
            )
        # Create the checkpoint folder if it does not exist
        if is_checkpoint:
            os.path.join(model_folder, "checkpoints")
            if os.path.exists(model_folder) is False:
                os.makedirs(model_folder)
                print(
                    f"Checkpoint folder does not exist."
                    f"Creating it: {model_folder}"
                )
        # Return the model folder
        return model_folder

    def get_model_path(
        self, config: ConfigType, is_checkpoint: bool = False
    ) -> str:
        """Method to get the path to the right model file,
        called when saving the model
        """

        # Reward model
        if isinstance(config, ConfigReward):
            model_name = self.get_model_name(config.model)
            model_folder = self.get_model_folder(
                config.model_folder,
                is_checkpoint,
            )
            model_path = os.path.join(model_folder, model_name)
            return model_path

    def check_model_path(
        self, config: ConfigType, is_checkpoint: bool = False
    ) -> Optional[str]:
        """Method to check if the model path exists,
        called when loadding the model
        """
        model_path = self.get_model_path(config, is_checkpoint)
        if os.path.exists(model_path) is False:
            if is_checkpoint:
                print(f"No previous checkpoint found at {model_path}")
            else:
                print(f"No previous model found at {model_path}")
            return None
        else:
            return model_path
