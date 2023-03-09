import json
import os

import deepspeed
import torch
from beartype import beartype
from beartype.typing import Optional, Iterable
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
from transformers import BartModel
from transformers import (
    BartTokenizer,
    BartConfig,
    AutoModel,
    AutoTokenizer,
)

from chatllama.rlhf.config import ConfigReward
from chatllama.rlhf.model_list import hf_models
from chatllama.rlhf.utils import TrainingStats

# TODO: Remove distillation from here


class RewardModel(torch.nn.Module):
    """Model to be trained to predict the reward for RL.
    or to be used as Critic in RL.

    Attributes:
        model (torch.nn.Module): Model to be used for the reward model
        tokenizer (torch.nn.Module): Tokenizer to be used for the reward model
        head (torch.nn.Module): Head to be used for the reward model
        config (ConfigReward): Config parameters for the reward model
        max_model_tokens (int): Maximum sequence length for the reward model

    Methods:
        forward: Forward pass of the model (used by the critic)
        save: Save the model
        load: Load the model
        get_reward: Get the reward for a given input (used by the reward model)
    """

    def __init__(self, config: ConfigReward) -> None:
        super().__init__()
        # load the model -- add here other models
        head_hidden_size = config.model_head_hidden_size
        if config.model in hf_models:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                padding_side="left",
                truncation_side="left",
            )
            self.model = AutoModel.from_pretrained(config.model)
            # galactica tokenizer eos_token is None
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "</s>"
                self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            head_dim = self.model.config.hidden_size
            if config.model.startswith("gpt2"):
                head_dim = self.model.config.n_embd
            self.head = torch.nn.Sequential(
                torch.nn.Linear(head_dim, head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden_size, 1),
                Rearrange("... 1 -> ..."),
            )
        elif config.model == "bart-base":
            bart_config = BartConfig.from_pretrained("facebook/bart-base")
            bart_config.max_position_embeddings = 2048 + 1024
            self.model = BartModel(bart_config)
            self.tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large",
                padding_side="left",
                truncation_side="left",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.head = torch.nn.Sequential(
                torch.nn.Linear(768, head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden_size, 1),
                Rearrange("... 1 -> ..."),
            )
        elif config.model == "longformer-base-4096":
            self.model = AutoModel.from_pretrained(
                "allenai/longformer-base-4096"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/longformer-base-4096",
                padding_side="left",
                truncation_side="left",
            )
            self.tokenizer.eos_token = self.tokenizer.pad_token
            self.head = torch.nn.Sequential(
                torch.nn.Linear(768, head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden_size, 1),
                Rearrange("... 1 -> ..."),
            )
        else:
            raise ValueError(f"model {config.model} not supported")
        # store config
        self.config = config
        if os.path.exists(config.model_folder) is False:
            os.mkdir(config.model_folder)
        else:
            self.load()
        # freeze model parameters (only train the head)
        for param in self.model.parameters():
            param.requires_grad = False
        # move model to device
        self.model.to(config.device)
        self.head.to(config.device)

    @beartype
    def parameters(
        self,
    ) -> Iterable[torch.nn.Parameter]:
        """Return the parameters of the reward model"""
        for p in self.model.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    @beartype
    def forward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate the sequence of rewards for the given output sequence
        what is the quality of the output sequence tokens?

        Args:
            output_sequence (torch.Tensor): The sequence of tokens to be
                evaluated
            output_sequence_mask (torch.Tensor): Mask for the attention

        Returns:
            torch.Tensor: Rewards for the given output sequence
        """
        output = self.model(
            output_sequence, attention_mask=output_sequence_mask
        )
        # What if the output_sequence is longer than the max context of
        # the model?
        rewards = self.head(output.last_hidden_state)
        if self.config.debug:
            print("RewardModel.forward")
            print("output_sequence.shape", output_sequence.shape)
            print("output_sequence", output_sequence)
            print("reward.shape", rewards.shape)
            print("reward", rewards)
        return rewards

    @beartype
    def get_reward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the reward for the given output sequence

        Args:
            output_sequence (torch.Tensor): The concatenation of initial input
                and actor output as tokens
            output_sequence_mask (torch.Tensor): Mask for the attention
        """
        rewards = self.forward(output_sequence, output_sequence_mask)
        print("rewards shape", rewards.shape)
        return rewards[:, -1]

    @beartype
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path

        Args:
            path (str): path to the model
        """
        if path is None:
            if self.config.model in hf_models:
                model_name = os.path.split(self.config.model)[-1]
            else:
                model_name = self.config.model
            path = os.path.join(self.config.model_folder, f"{model_name}.pt")
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
                print(
                    f"Model folder does not exist. Creating it,"
                    f"and returning without loading the model:\n{path}"
                )
                return
        # load the model
        if os.path.exists(path) is False:
            print(
                f"Warning, Impossible to load the model:\n{path}\n"
                f"No previous checkpoint found."
            )
            return
        model_dict = torch.load(path)
        self.model.load_state_dict(model_dict["model"])
        self.head.load_state_dict(model_dict["head"])

    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path

        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            if self.config.model in hf_models:
                model_name = os.path.split(self.config.model)[-1]
            else:
                model_name = self.config.model
            path = os.path.join(self.config.model_folder, f"{model_name}.pt")
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
        torch.save(
            {"model": self.model.state_dict(), "head": self.head.state_dict()},
            path,
        )


# just to keep namings consistent
CriticModel = RewardModel


class RewardDataset(Dataset):
    """Dataset class for the reward model
    read a json file with the following format:
    [
        {
            "user_input": "...",
            "completion": "...",
            "score": ...
        },
        ...
    ]
    Where:
        user_input: the initial input of the user
        completion: the completion generated by the model
        score: the score given by the user to the completion (or by the LLM)
    """

    def __init__(self, path: str) -> None:
        print(f"Loading dataset from {path}")
        with open(path, "r") as f:
            self.data = list(json.load(f))
        print(f"Loaded {len(self.data)} samples")

    def __getitem__(self, idx: int):
        user_input = self.data[idx]["user_input"]
        completion = self.data[idx]["completion"]
        score = float(self.data[idx]["score"])
        item = (user_input + " " + completion, score)
        return item

    def __len__(
        self,
    ):
        return len(self.data)


class RewardTrainer:
    """Reward class to train the reward model

    Args:
        config (ConfigModel): Config parameters for the model

    Attributes:
        model (RewardModel): Reward model
        config (ConfigModel): Config parameters for the model
        optimizer (torch.optim): Optimizer for the model
        loss (torch.nn): Loss function for the model

    Methods:
        train: Train the reward model
        generate_user_input: Generate the user input for the LLM to evaluate a
            couple, (user_input, completion) and assing a score
        distill: Parse the dataset and assign scores using LLMs
    """

    def __init__(self, config: ConfigReward) -> None:
        # load the model, optimizer, loss function and config
        self.model = RewardModel(config)
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )
        self.loss_function = torch.nn.MSELoss()

        # check checkpoint, datasets and other data
        if not os.path.exists("./models"):
            os.mkdir("./models")
        self.training_stats = TrainingStats()
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataloaders
        self.train_dataset = RewardDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            self.eval_dataset = RewardDataset(config.validation_dataset_path)
            self.validation_dataloader = DataLoader(
                self.eval_dataset, batch_size=config.batch_size
            )

        # initialize deepspeed
        self.model_engine = None
        if config.deepspeed_enable is True:
            if config.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if os.path.exists(config.deepspeed_config_path) is False:
                raise ValueError(
                    f"DeepSpeed config path {config.deepspeed_config_path}"
                    f"does not exist"
                )
        if self.config.deepspeed_enable:
            (
                self.model_engine,
                self.optimizer,
                self.train_dataloader,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.model,
                model_parameters=self.model.parameters(),
                training_data=self.train_dataloader,
                config=self.config.deepspeed_config_path,
            )

    def train(
        self,
    ) -> None:
        """Train the reward model"""
        print("Start Training the Reward Model")
        # get config parameters
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device
        iteration_per_print = self.config.iteration_per_print

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # traing loop
        for epoch in range(epochs):
            self.model.train()
            for i, inputs in enumerate(self.train_dataloader):
                input_text = inputs[0]
                score = inputs[1]

                # tokenizer (placed here instead of dataset class)
                input_tokens = self.model.tokenizer(
                    input_text, padding=True, truncation=True
                )

                # TODO: check on the length of the input tokens if they are
                # too many it can create problems
                output = torch.as_tensor(
                    score, dtype=torch.float32, device=device
                )

                # forward pass
                if self.config.deepspeed_enable:
                    est_output = self.model_engine(
                        torch.as_tensor(
                            input_tokens["input_ids"], device=device
                        ),
                        torch.as_tensor(
                            input_tokens["attention_mask"], device=device
                        ),
                    )[:, -1]
                else:
                    est_output = self.model.get_reward(
                        torch.as_tensor(
                            input_tokens["input_ids"], device=device
                        ),
                        torch.as_tensor(
                            input_tokens["attention_mask"],
                            device=device,
                        ),
                    )

                loss = self.loss_function(est_output, output)
                self.training_stats.training_loss.append(loss.item())

                # backward pass
                if self.config.deepspeed_enable:
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # print progress
                if i % iteration_per_print == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss.item()}"
                    )
                    print(
                        "prediction",
                        est_output.cpu().detach().numpy(),
                        "target",
                        score.cpu().numpy(),
                    )
            if self.validation_flag:
                self.model.eval()
                for i, (text, score) in enumerate(self.validation_dataloader):
                    # forward pass
                    input_tokens = self.model.tokenizer(
                        text, return_tensors="pt", padding=True
                    )
                    input_tokens = input_tokens.to(device)
                    # TODO: check on the length of the input tokens if they are
                    # too many it can create problems
                    output = torch.tensor(score, dtype=torch.float32).to(
                        device
                    )
                    est_output = self.model.get_reward(
                        input_tokens["input_ids"],
                        input_tokens["attention_mask"],
                    )
                    loss = self.loss_function(est_output, output)
                    self.training_stats.validation_loss.append(loss.item())

                    # print progress
                    if i % iteration_per_print == 0:
                        print(
                            f"Epoch: {epoch+1}/{epochs}, "
                            f"Iteration: {i+1}/{n_iter}, "
                            f"Validation Loss: {loss.item()}"
                        )
        print("Saving the model...")
        self.model.save()
