import yaml
import os
from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Optional


@dataclass
class ConfigReward:
    """Config parameters for the reward model

    Attributes:
        device (torch.device): Device to be used for the reward model
        model (str): Model to be used for the reward model
        model_head_hidden_size (int): Hidden size of the reward model head
        model_folder (str): Path to the folder where model are stored (used
            to load / store finetuned model)

        train_dataset_path (Optional[str]): Path to the training dataset.
            Default to None. To be specified only for the reward model trainig.
        validation_dataset_path (Optional[str]): Path to the validation
            dataset. Default to None. To be specified only for the reward
            model trainig.
        batch_size (Optional[int]): Batch size to train the reward model.
            Default to None. To be specified only for the reward model
            trainig.
        epochs (Optional[int]): Number of epochs to train the reward model.
            Default to None. To be specified only for the reward model
            trainig.
        iteration_per_print (Optional[int]): Number of iterations to print
            the training loss. Default to None. To be specified only for the
            reward model trainig.
        lr (Optional[float]): Learning rate for the reward model. Default to
            None. To be specified only for the reward model distillation.
        llm_enable (bool): Enable reward model distillation. Default to True.
            Disable it if you dont have an API key.
        llm_model (Optional[str]): Model to be used for the reward model
            distillation. Default to "text-davinci-003".
        llm_temperature (Optional[float]): Temperature for the reward model
            distillation. Default to 0.9.
        llm_max_tokens (Optional[int]): Max tokens for the reward model
            distillation. Default to 64.
        deepspeed_enable (bool): Enable deepspeed for the reward model
            training. Default to False.
        deepspeed_config_path (str): Path to the deepspeed config file.
            Default to None.
        debug (bool): enable prints for Debugging
    """

    device: torch.device
    model: str
    model_head_hidden_size: int
    model_folder: str
    train_dataset_path: Optional[str] = None
    validation_dataset_path: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    iteration_per_print: Optional[int] = None
    lr: Optional[float] = None
    llm_enable: Optional[bool] = False
    llm_model: Optional[str] = "text-davinci-003"
    llm_temperature: Optional[float] = 0.9
    llm_max_tokens: Optional[int] = 64
    deepspeed_enable: bool = False
    deepspeed_config_path: Optional[str] = None

    debug: bool = False


@dataclass
class ConfigActor:
    """Config parameters for models

    Attributes:
        model (str): Model to be used for the actor
        model_path (str): Path to the model (used to load and store the model)
        checkpoint_folder (str): Path to the folder where checkpoints are
            stored.
        tokenizer_folder (str): Path to the folder where tokenizer are stored
        train_dataset_path (str): Path to the training dataset
        validation_dataset_path (Optional[str]): Path to the validation dataset
        froze_embeddings (bool): Froze embeddings for the actor
        use_fairscale (bool): Use fairscale module for the actor instead of
            pytorch native modules.
        max_sequence_length (int): Max sequence length for the actor
        max_tokens (int): Max tokens for actor generation
        temperature (float): Temperature for the actor
        batch_size (int): Batch size to train the actor
        iteration_per_print (int): Number of iterations to print the
            training loss
        lr (float): Learning rate for the actor
        epochs (int): Number of epochs to train the actor
        deepspeed_enable (bool): Enable deepspeed for the actor.
            Default to False.
        deepspeed_config_path (str): Path to the deepspeed config file.
            Default to None.
        device (torch.device): Device to be used for the actor
        debug (bool): Enable prints for debugging
    """

    model: str
    model_path: str
    checkpoint_folder: str
    tokenizer_folder: str
    train_dataset_path: str
    validation_dataset_path: Optional[str]
    froze_embeddings: bool
    use_fairscale: bool
    max_sequence_length: int
    max_tokens: int
    temperature: float
    batch_size: int
    iteration_per_print: int
    lr: float
    epochs: int

    deepspeed_enable: bool
    deepspeed_config_path: Optional[str]

    device: torch.device
    debug: bool = False


@dataclass
class ConfigTrainer:
    """Config parameters for the trainer, used to configure the reinforcement
    learning training loop

    Attributes:
        actor_lr (float): Learning rate for the actor when training with
            reinforcement learning
        critic_lr (float): Learning rate for the critic when training with
            reinforcement learning
        actor_eps_clip (float): Epsilon clip for the actor
        critic_eps_clip (float): Epsilon clip for the critic
        beta_s (float): Beta for the actor and critic
        examples_path (str): Path to the examples dataset
        num_episodes (int): Number of episodes, each episodes consist of
            a number of timesteps that are used to generate examples
            stored in the memory buffer.
        max_timesteps (int): Max timesteps for the actor and critic.
            for each timestep a set of examples are sampled and used to
            generate a completion and a reward.
        update_timesteps (int): Number of timesteps to update the actor and
            critic
        num_examples (int): Number of examples to generate for the actor
            and critic. For each iteration of timestep, num_examples are
            sampled from the prompt dataset, processed and stored in the
            memory buffer.
        batch_size (int): Batch size to train the actor and critic.
            This batch is used to aggregate the memory from the memory buffer
            for the actual training of the actor and critic models.
        epochs (int): Number of epochs to train the actor and critic.
        update_checkpoint (int): Number of timesteps to update the checkpoint
        checkpoint_folder (str): Folder to store the checkpoints
    """

    actor_lr: int
    critic_lr: int
    actor_eps_clip: float
    critic_eps_clip: float
    beta_s: float
    examples_path: str
    num_episodes: int
    max_timesteps: int
    update_timesteps: int
    num_examples: int
    batch_size: int
    epochs: int
    update_checkpoint: int
    checkpoint_folder: str

    device: torch.device
    debug: bool = False


class Config:
    """Store the config parameters for the whole pipeline

    Args:
        trainer_dict (Optional[Dict]): Dictionary with the config parameters
            for the trainer. Default to None. If None, the config.yaml is
            used.
        actor_dict (Optional[Dict]): Dictionary with the config parameters
            for the actor. Default to None. If None, the config.yaml is
            used.
        critic_dict (Optional[Dict]): Dictionary with the config parameters
            for the critic. Default to None. If None, the config.yaml is
            used.
        reward_dict (Optional[Dict]): Dictionary with the config parameters
            for the reward. Default to None. If None, the config.yaml is
            used.
        device (Optional[torch.device]): Device to be used for the actor
            and critic. Default to None. If None, the device available is
            used.
        debug (Optional[bool]): Enable prints for debugging. Default to False.

    Attributes:
        trainer (ConfigTrainer): Config parameters for the trainer
        actor (ConfigActor): Config parameters for the actor
        critic (ConfigCritic): Config parameters for the critic
        reward (ConfigReward): Config parameters for the reward
    """

    @beartype
    def __init__(
        self,
        path: str,
        device: Optional[torch.device] = None,
        debug: Optional[bool] = False,
    ) -> None:

        # if not specified use the device available
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise ValueError("No GPU available")
            print(f"Current device used :{str(device)}")

        if path is None or os.path.exists(path) is False:
            raise ValueError("Path to the config.yaml is not valid")

        # Read the config from yaml
        with open(path, "r") as c:
            config = yaml.safe_load(c)

        trainer_dict = config["trainer_config"]
        actor_dict = config["actor_config"]
        critic_dict = config["critic_config"]
        reward_dict = config["reward_config"]

        # Trainer Config
        trainer_dict["device"] = device
        trainer_dict["debug"] = debug
        self.trainer = ConfigTrainer(**trainer_dict)
        # Actor Config
        actor_dict["device"] = device
        actor_dict["debug"] = debug
        self.actor = ConfigActor(**actor_dict)
        # Critic Config
        critic_dict["device"] = device
        critic_dict["debug"] = debug
        self.critic = ConfigReward(**critic_dict)
        # Reward Config
        reward_dict["device"] = device
        reward_dict["debug"] = debug
        self.reward = ConfigReward(**reward_dict)
