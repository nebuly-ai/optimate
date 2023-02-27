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
        model (str): Model to be used for the reward model
        model_folder (str): Path to the folder where model are stored (used
            to load / store finetuned model)
        device (torch.device): Device to be used for the reward model
        model_head_hidden_size (int): Hidden size of the reward model head
        debug (bool): enable prints for Debugging
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
        llm_model (Optional[str]): Model to be used for the language model
            (LLM). Default to None.
        llm_max_tokens (Optional[int]): Max tokens for the LLM. Default to
            None.
        llm_temperature (Optional[float]): Temperature for the LLM. Default
            to None.
        deepspeed_enable (bool): Enable deepspeed for the reward model
            training. Default to False.
        deepspeed_config_path (str): Path to the deepspeed config file.
            Default to None.
    """

    model: str
    model_folder: str
    device: torch.device
    model_head_hidden_size: int
    debug: bool
    train_dataset_path: Optional[str] = None
    validation_dataset_path: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    iteration_per_print: Optional[int] = None
    lr: Optional[float] = None
    llm_model: Optional[str] = None
    llm_max_tokens: Optional[int] = None
    llm_temperature: Optional[float] = None
    deepspeed_enable: bool = False
    deepspeed_config_path: Optional[str] = None


@dataclass
class ConfigActor:
    """Config parameters for models

    Attributes:
        model (str): Model to be used for the actor
        model_folder (str): Path to the folder where model are stored (used
            to load / store finetuned model)
        max_tokens (int): Max tokens for the actor
        temperature (float): Temperature for the actor
        device (torch.device): Device to be used for the actor
        lr (float): Learning rate for the actor
        iteration_per_print (int): Number of iterations to print the
            training loss
        batch_size (int): Batch size to train the actor
        epochs (int): Number of epochs to train the actor
        debug (bool): Enable prints for debugging
        train_dataset_path (str): Path to the training dataset
        validation_dataset_path (Optional[str]): Path to the validation dataset
        deepspeed_enable (bool): Enable deepspeed for the actor.
            Default to False.
        deepspeed_config_path (str): Path to the deepspeed config file.
            Default to None.
    """

    model: str
    model_folder: str
    tokenizer_folder: str
    max_tokens: int
    temperature: float
    device: torch.device
    lr: float
    iteration_per_print: int
    batch_size: int
    epochs: int
    debug: bool
    train_dataset_path: str
    validation_dataset_path: Optional[str] = None
    deepspeed_enable: bool = False
    deepspeed_config_path: Optional[str] = None


@dataclass
class ConfigTrainer:
    """Config parameters for the trainer, used to configure the reinforcement
    learning training loop

    Attributes:
        update_timesteps (int): Number of timesteps to update the actor
            and critic. Every time update_timesteps timesteps are collected,
            the training loop for the actor and critic is executed using the
            memory buffer to learn the policy.
        temperature (float): Temperature for the actor and critic
        max_seq_len (int): Max sequence length for the actor and critic
        num_examples (int): Number of examples to generate for the actor
            and critic. For each iteration of timestep, num_examples are
            sampled from the prompt dataset, processed and stored in the
            memory buffer.
        actor_lr (float): Learning rate for the actor when training with
            reinforcement learning
        critic_lr (float): Learning rate for the critic when training with
            reinforcement learning
        num_episodes (int): Number of episodes, each episodes consist of
            a number of timesteps that are used to generate examples
            stored in the memory buffer.
        max_timesteps (int): Max timesteps for the actor and critic.
            for each timestep a set of examples are sampled and used to
            generate a completion and a reward.
        batch_size (int): Batch size to train the actor and critic.
            This batch is used to aggregate the memory from the memory buffer
            for the actual training of the actor and critic models.
        epochs (int): Number of epochs to train the actor and critic.
        actor_eps_clip (float): Epsilon clip for the actor
        critic_eps_clip (float): Epsilon clip for the critic
        beta_s (float): Beta for the actor and critic
        update_checkpoint (int): Number of timesteps to update the checkpoint
        llm_model_id (str): Model id for the llm
        llm_max_tokens (int): Max tokens for the llm
        llm_temperature (float): Temperature for the llm
        device (torch.device): Device to be used for the actor and critici
        checkpoint_folder (str): Folder to store the checkpoints while training
        debug (bool): Enable prints for debugging
        deepspeed_enable (bool): Enable deepspeed for the actor and critic.
            Default to False.
        deepspeed_config_path (str): Path to the deepspeed config file.
            Default to None.
    """

    update_timesteps: int
    num_examples: int
    actor_lr: float
    critic_lr: float
    num_episodes: int
    max_timesteps: int
    examples_path: str
    batch_size: int
    epochs: int
    actor_eps_clip: float
    critic_eps_clip: float
    beta_s: float
    update_checkpoint: int
    llm_model_id: str
    llm_max_tokens: int
    llm_temperature: float
    device: torch.device
    checkpoint_folder: str
    debug: bool


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
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"Current device used:{str(device)}")

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
