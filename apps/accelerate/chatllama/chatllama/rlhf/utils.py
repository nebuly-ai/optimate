import json
import os
import re
import sys

import deepspeed
import logging
import torch
from beartype import beartype
from beartype.typing import Union, Tuple, Optional
from loguru import logger
from plotly import graph_objects as go
from transformers import AutoTokenizer

from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)
from chatllama.rlhf.model_list import hf_models, llama_models


ConfigType = Union[Config, ConfigActor, ConfigCritic, ConfigReward]


class Singleton:
    """Singleton class to ensure only one instance of a class is created"""

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance


class LogMessages(Singleton):
    """
    Class to handle all logging messages
    """

    def __init__(
        self,
    ):

        set_global_logging_level()

        self.log_config = {
            "handlers": [
                {
                    "sink": sys.stdout,
                    "format": (
                        "<level> {time:YY-MM-DD HH:mm:ss.S}"
                        " | "
                        "{message} </level>"
                    ),
                    "colorize": True,
                    "level": "INFO",
                },
                {
                    "sink": "file.log",
                    "serialize": True,
                },
            ],
        }
        logger.configure(**self.log_config)

        # flag to enable multi gpu logging
        self.is_multi_gpu = False

        # rank to print when multi gpu
        self.log_rank = -1

        # local rank
        self.local_rank = -2

    def setup_logger(self, accelerate_enable: bool, deepspeed_enable: bool):
        """Setup logger for multi gpu training

        Args:
            accelerate_enable (bool): flag to signal if using accelerate
            deepspeed_enable (bool): flag to signal if using deepspeed
        """
        if accelerate_enable or deepspeed_enable:
            self.is_multi_gpu = True
            self.log_rank = 0
            self.local_rank = int(os.environ.get("RANK", "0"))
        else:
            self.is_multi_gpu = False

    def error(self, error_type, text: str):
        """Log error message

        Args:
            error_type (Exception): type of error to raise
            text (str): error message to log
        """
        if self.is_multi_gpu:
            if self.local_rank == self.log_rank:
                logger.error(f"[Rank {self.local_rank}] {text}")
        else:
            logger.error(text)
        return error_type("")

    def warning(self, text: str):
        """Log warning message

        Args:
            text (str): warning message to log
        """
        if self.is_multi_gpu:
            if self.local_rank == self.log_rank:
                logger.warning(f"[Rank {self.local_rank}] {text}")
        else:
            logger.warning(text)

    def info(self, text: str):
        """Log info message

        Args:
            text (str): info message to log
        """
        if self.is_multi_gpu:
            if self.local_rank == self.log_rank:
                logger.info(f"[Rank {self.local_rank}] {text}")
        else:
            logger.info(text)

    def success(self, text: str):
        """Log success message

        Args:
            text (str): success message to log
        """
        if self.is_multi_gpu:
            if self.local_rank == self.log_rank:
                logger.success(f"[Rank {self.local_rank}] {text}")
        else:
            logger.success(text)
            
    def debug(self, text: str):
        """Log debug message
        
        Args:
            text (str): debug message to log
        """
        
        if self.is_multi_gpu:
            if self.local_rank == self.log_rank:
                logger.debug(f"[Rank {self.local_rank}] {text}")
        else:
            logger.debug(text)


# To control logging level for various modules used in the application:
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a
    prefix.
    It needs to be invoked after the modules have been loaded so that their
    loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional.
        Default is logging.ERROR
        - prefices: list of one or more str prefices to match
        (e.g. ["transformers", "torch"]). Optional. Default is `[""]`
        to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

    # force deepspeed
    deepspeed.utils.logging.logger.setLevel(level)


# configure logger and logging level
my_logger = LogMessages()


@beartype
def get_multi_gpu_flags(
    config: ConfigType,
) -> Tuple[bool, bool, Optional[str]]:
    """Setup for multi-gpu training

    Args:
        config (ConfigType): Config object

    Returns:
        Tuple[bool, bool, Optional[str]]: Tuple of flags for multi-gpu training
    """

    # flags for training
    if isinstance(config, Config):
        accelerate_enable = config.trainer.accelerate_enable
        deepspeed_enable = config.trainer.deepspeed_enable
        deepspeed_config_path = config.trainer.deepspeed_config_path
    else:
        accelerate_enable = config.accelerate_enable
        deepspeed_enable = config.deepspeed_enable
        deepspeed_config_path = config.deepspeed_config_path

    # check consistency of flags
    if accelerate_enable and deepspeed_enable:
        raise my_logger.error(
            ValueError,
            (
                "Both DeepSpeed and Accelerate are enabled"
                + "Please choose one of them."
            ),
        )

    # check deepspeed config
    if deepspeed_enable:
        if deepspeed_config_path is None:
            raise my_logger.error(
                ValueError,
                "DeepSpeed config path is None, but deepspeed is enabled",
            )
        if os.path.exists(deepspeed_config_path) is False:
            raise my_logger.error(
                ValueError,
                f"DeepSpeed config path"
                f" {deepspeed_config_path} "
                f"does not exist",
            )

    # setup the logger consequently
    my_logger.setup_logger(accelerate_enable, deepspeed_enable)

    return accelerate_enable, deepspeed_enable, deepspeed_config_path


def load_tokenizer(config: ConfigType):
    """Load the tokenizer from the model name
    placed in utils to avoid circular imports from dataset and model class

    Args:
        config (ConfigType): config object
    """

    # disable tokenizer parallelization (Avoid warnings in HF)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config.model in hf_models:

        # load the tokenizer from HF
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            padding_side="left",
            padding=True,
            truncation=True,
            model_max_length=config.max_sequence_length,
        )

        # add eos token if not present
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
            tokenizer.eos_token_id = 2  # OPT eos-token-id

        # add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    elif config.model in llama_models:

        if not isinstance(config, ConfigActor):
            raise my_logger.error(
                ValueError,
                "LLaMA models can only be used as actor",
            )

        # llama module might not be present when HF models are used
        from chatllama.llama_model import (
            load_tokenizer,
        )  # noqa

        tokenizer = load_tokenizer(config.tokenizer_path)
    return tokenizer


class TrainingStats:
    """Training statistics

    Attributes:
        training_loss (List): List of training losses
        training_accuracy (List): List of training accuracies
        value_loss (List): List of value losses
        validation_loss (List): List of validation losses
        validation_accuracy (List): List of validation accuracies
    """

    def __init__(self, path: str):
        """Initialize the training stats

        Args:
            path (str): Path to save the stats
        """
        self.training_loss = []
        self.training_accuracy = []
        self.value_loss = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.path = path

    def plot(self):
        """Plot the training statistics using plotly"""
        fig = go.Figure()
        if len(self.training_loss) > 0:
            fig.add_trace(
                go.Scatter(y=self.training_loss, name="Training loss")
            )
        if len(self.training_accuracy) > 0:
            fig.add_trace(
                go.Scatter(y=self.training_accuracy, name="Training accuracy")
            )
        if len(self.value_loss) > 0:
            fig.add_trace(go.Scatter(y=self.value_loss, name="Value loss"))
        if len(self.validation_loss) > 0:
            fig.add_trace(
                go.Scatter(y=self.validation_loss, name="Validation loss")
            )
        if len(self.validation_accuracy) > 0:
            fig.add_trace(
                go.Scatter(
                    y=self.validation_accuracy, name="Validation accuracy"
                )
            )
        fig.update_layout(
            showlegend=True, xaxis_type="log", xaxis_title="steps"
        )
        fig.show()

    def save(
        self,
    ):
        """Save the stats"""
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                stats = json.load(f)
            stats["training_loss"].extend(self.training_loss)
            stats["training_accuracy"].extend(self.training_accuracy)
            stats["value_loss"].extend(self.value_loss)
            stats["validation_loss"].extend(self.validation_loss)
            stats["validation_accuracy"].extend(self.validation_accuracy)
        else:
            stats = {
                "training_loss": self.training_loss,
                "training_accuracy": self.training_accuracy,
                "value_loss": self.value_loss,
                "validation_loss": self.validation_loss,
                "validation_accuracy": self.validation_accuracy,
            }
        with open(self.path, "w") as f:
            json.dump(stats, f, indent=4)

    def load(
        self,
    ):
        """Load the stats"""
        with open(self.path, "r") as f:
            stats = json.load(f)
        self.training_loss = stats["training_loss"]
        self.training_accuracy = stats["training_accuracy"]
        self.value_loss = stats["value_loss"]
        self.validation_loss = stats["validation_loss"]
        self.validation_accuracy = stats["validation_accuracy"]

    def clear(
        self,
    ):
        """Clear the stats"""
        self.training_loss = []
        self.training_accuracy = []
        self.value_loss = []
        self.validation_loss = []
        self.validation_accuracy = []
        if os.path.exists(self.path):
            os.remove(self.path)


class ConversationLog:
    """Save the conversation:
    (user input, model output, rewards and learn_counter)
    during the RL training loop.
    """

    def __init__(self, path: str):
        self.conversation = []
        self.path = path
        if self.path is None:
            self.path = "./convesation_log.json"

    @beartype
    def append(
        self,
        user_input: str,
        model_output: str,
        reward: float,
        learn_counter: int,
    ):
        """Add a conversation to the log

        Args:
            user_input (str): User input / initial prompt
            model_output (str): Completion of the LLM model
            reward (float): Reward of the reward model assigned to the output
            learn_counter (int): Number of the learning iteration to
                distinguish the conversations that happens at different
                points of the training loopt
        """
        self.conversation.append(
            {
                "user_input": user_input,
                "model_output": model_output,
                "reward": reward,
                "learn_counter": learn_counter,
            }
        )

    def save(self):
        """Save the conversation log"""
        my_logger.info("Saving conversations log")
        # load previous conversations - commented out
        # if os.path.exists(self.path):
        #     with open(self.path, "r") as f:
        #         conversation = json.load(f)
        #     self.conversation.extend(conversation)
        self.conversation = sorted(
            self.conversation, key=lambda x: float(x["learn_counter"])
        )
        with open(self.path, "w") as f:
            json.dump(self.conversation, f, indent=4)

    def load(self):
        """Load the conversation log"""
        with open(self.path, "r") as f:
            self.conversation = json.load(f)

    def clear(self):
        """Clear the conversation log"""
        my_logger.info("Clearing conversations log")
        self.conversation = []
        # remove the file in path exists
        if os.path.exists(self.path):
            os.remove(self.path)

    def show(self, current_iteration: int = None):
        """Show the conversation log

        Args:
            current_iteration (int): Current iteration of the training loop,
                if not None, print only the conversations that happened at
                <current_iteration>
        """

        # TODO: use logger to show messages...
        for i, c in enumerate(self.conversation):
            if current_iteration is None:
                print(
                    f"##########################################\n"
                    f"Conversation {i} at learn_counter "
                    f"{c['learn_counter']}\n"
                    f"##########################################\n"
                    f"## User Input:\n\n{c['user_input']}\n\n"
                    f"## Model Output:\n\n{c['model_output']}\n\n"
                    f"## Reward: {c['reward']}\n\n"
                )
            else:
                if current_iteration == c["learn_counter"]:
                    print(
                        f"##########################################\n"
                        f"Conversation {i} at learn_counter "
                        f"{c['learn_counter']}\n"
                        f"##########################################\n"
                        f"## User Input:\n\n{c['user_input']}\n\n"
                        f"## Model Output:\n\n{c['model_output']}\n\n"
                        f"## Reward: {c['reward']}\n\n"
                    )


class IgnoreLabelsWrapper(torch.nn.Module):
    """Wrapper to ignore labels arguments when using lora models
    with AutoModel HF models.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Remove labels, which are unused by the base model
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
