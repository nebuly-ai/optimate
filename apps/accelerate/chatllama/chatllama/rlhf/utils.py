import json
import os
import sys

import torch
from beartype import beartype
from beartype.typing import Union, Optional
from loguru import logger
from plotly import graph_objects as go

from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)


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
        print("Saving conversations log")
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                conversation = json.load(f)
            self.conversation.extend(conversation)
        self.conversation = sorted(
            self.conversation, key=lambda x: float(x["learn_counter"])
        )
        with open(self.path, "w") as f:
            json.dump(self.conversation, f, indent=4)

    def load(self):
        with open(self.path, "r") as f:
            self.conversation = json.load(f)

    def clear(self):
        print("Clearing conversations log")
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


ConfigType = Union[Config, ConfigActor, ConfigCritic, ConfigReward]


class Singleton:
    __instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance


class LogMessages(Singleton):
    config = None
    
    def __init__(self, config: Optional[ConfigType] = None):
        if config is not None:
            self.config = config
        if self.config is None:
            # the config for the logger has not been set yet.
            # the defualt beahviour should be single vanilla single GPU
            pass
        self.log_config = {
            "handlers": [
                {
                    "sink": sys.stdout,
                    "format": "<level> {time:YYYY-MM-DD HH:mm:ss.SSS} | {message} </level>",
                    "colorize": True,
                    "level": "INFO"
                },
                {
                    "sink": "file.log",
                    "serialize": True,
                },
            ],
        }
        logger.configure(**self.log_config)
        
    def error(self, error_type, text: str):
        logger.error(text)
        return error_type(text)
    
    def warning(self, text: str):
        logger.warning(text)
    
    def info(self, text: str):
        logger.info(text)
        
    def success(self, text: str):
        logger.success(text)
    