import json
import shutil
import os

import deepspeed
import torch
from accelerate import Accelerator
from beartype import beartype
from beartype.typing import Iterable, Tuple
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
)

from chatllama.rlhf.config import ConfigReward
from chatllama.rlhf.model_list import hf_models
from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.utils import TrainingStats


class RewardModel(torch.nn.Module):
    """Model to be trained to predict the reward for RL.
    or to be used as Critic in RL. It is a Language Model with a head
    that predicts the reward (a scalar) for a given sequence of tokens.

    Attributes:
        model (torch.nn.Module): Model to be used for the reward model
        tokenizer (torch.nn.Module): Tokenizer to be used for the reward model
        head (torch.nn.Module): Head to be used for the reward model
        config (ConfigReward): Config parameters for the reward model

    Methods:
        load_tokenizer: Load the tokenizer for the reward model
        forward: Forward pass of the model (used by the critic)
        save: Save the model
        load: Load the model
        get_reward: Get the reward for a given input (used by the reward model)
        parameters: Return the parameters of the reward model

    """

    def __init__(self, config: ConfigReward) -> None:
        super().__init__()

        # store config
        self.config = config

        # initialize the self.model
        head_hidden_size = config.model_head_hidden_size
        if config.model in hf_models:
            self.tokenizer = self.load_tokenizer(config)
            self.model = AutoModel.from_pretrained(config.model)
            head_dim = self.model.config.hidden_size
            if config.model.startswith("gpt2"):
                head_dim = self.model.config.n_embd
            self.head = torch.nn.Sequential(
                torch.nn.Linear(head_dim, head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden_size, 1),
                Rearrange("... 1 -> ..."),
            )
        else:
            raise ValueError(f"Model {config.model} not supported")

        # load the model
        self.load()

        # freeze model parameters (only train the head)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # move model to device
        self.model.to(config.device)
        self.head.to(config.device)

    @staticmethod
    def load_tokenizer(config: ConfigReward):
        # load tokenizer from HF
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
            tokenizer.eos_token_id = 2  # OPT  eos token id

        # add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @beartype
    def load(self) -> None:
        """Load the model from the path"""
        # look for a pretrained model
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=False,
            current_epoch=None,
        )

        # check if the model exists
        if path is not None:

            # load the model from the path
            print("Loading ...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict["model"])
            self.head.load_state_dict(model_dict["head"])

    @beartype
    def save(self) -> None:
        """Save the model to the path"""
        # get the path to save the model
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config,
            is_checkpoint=False,
            current_epoch=None,
        )

        # save the model
        print(f"Saving model to {path} ...")
        torch.save(
            {"model": self.model.state_dict(), "head": self.head.state_dict()},
            path,
        )

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
        if output_sequence.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"Output sequence is too long: {output_sequence.shape[1]}"
                f" > {self.config.max_sequence_length}"
            )
        rewards = self.forward(output_sequence, output_sequence_mask)
        return rewards[:, -1]


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
        item = (user_input + completion, score)
        return item

    def __len__(
        self,
    ):
        return len(self.data)


class RewardTrainer:
    """Class to train the reward model

    Args:
        config (ConfigModel): Config parameters for the model

    Attributes:
        model (RewardModel): Reward model
        config (ConfigModel): Config parameters for the model
        optimizer (torch.optim): Optimizer for the model
        loss_function (torch.nn): Loss function for the model
        validation_flag (bool): Flag to indicate if the validation dataset
            is available
        train_dataset (RewardDataset): Dataset for training
        validation_dataset (RewardDataset): Dataset for validation
        train_dataloader (DataLoader): Dataloader for training
        validation_dataloader (DataLoader): Dataloader for validation
        scheduler (torch.optim.lr_scheduler): Scheduler for the optimizer
        training_stats (List[Dict]): List of dictionaries with the training
            statistics
        model_engine (ModelEngine): Model engine to train the model
            using deepspeed
        accelerator (Accelerator): Accelerator to train the model using
            accelerate by HF.


    Methods:
        train: Train the reward model
        save_checkpoints: Save the checkpoints of the model
        load_checkpoints: Load the checkpoints of the model
    """

    def __init__(self, config: ConfigReward) -> None:

        # save the config
        self.config = config

        # load the model
        self.reward = RewardModel(config)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.reward.parameters(), lr=config.lr
        )

        # loss function
        self.loss_function = torch.nn.MSELoss()

        # check validation dataset
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataset and dataloaders
        self.train_dataset = RewardDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            self.eval_dataset = RewardDataset(config.validation_dataset_path)
            self.validation_dataloader = DataLoader(
                self.eval_dataset, batch_size=config.batch_size
            )

        # intilize scheduler - learning rate will drop to 10% of the initial
        # value
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_dataset) // config.batch_size,
            T_mult=1,
            eta_min=config.lr * 0.1,
            last_epoch=-1,
        )

        # initialize training stats
        stats_path = ModelLoader.get_training_stats_path(config)
        self.training_stats = TrainingStats(stats_path)

        # consistency check between accelerate and deepspeed
        if config.accelerate_enable and config.deepspeed_enable:
            raise ValueError(
                "Both DeepSpeed and Accelerate are enabled for the Reward."
                "Please choose one of them."
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
            (
                self.model_engine,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = deepspeed.initialize(
                args=None,
                model=self.reward,
                model_parameters=self.reward.parameters(),
                training_data=self.train_dataset,
                config=self.config.deepspeed_config_path,
            )
            print("Training with DeepSpeed")

        # initialize accelerate
        self.accelerator = None
        if config.accelerate_enable is True:
            self.accelerator = Accelerator()
            (
                self.reward,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.reward,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            )
            print("Training with Accelerate")

    @beartype
    def save_checkpoint(
        self,
        current_epoch: int,
        current_step: int,
        max_epochs: int,
        max_steps: int,
    ) -> None:
        """Save the checkpoints of the model

        Args:
            current_epoch (int): Current epoch
            current_step (int): Current step
            max_epochs (int): Maximum number of epochs
            max_steps (int): Maximum number of steps
        """

        print(
            f"Saving checkpoint for epoch {current_epoch + 1}, "
            f" step {current_step} ..."
        )

        # get the path to save the checkpoint
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=current_epoch,
            current_step=current_step,
            max_epochs=max_epochs,
            max_steps=max_steps,
        )

        # remove the checkpoint if it already exists
        if os.path.exists(path):
            if self.config.deepspeed_enable:
                shutil.rmtree(path)
            else:
                os.remove(path)

        # save the checkpoint
        if self.config.deepspeed_enable:
            client_state = {
                "epoch": current_epoch,
                "step": current_step,
            }
            self.model_engine.save_checkpoint(path, client_state=client_state)
        else:
            torch.save(
                {
                    "state_dict": self.reward.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "training_stats": self.training_stats,
                    "epoch": current_epoch,
                    "step": current_step,
                },
                path,
            )

    @beartype
    def load_checkpoint(
        self,
    ) -> Tuple[int, int]:
        """Load the checkpoints of the model

        Returns:
            Tuple[int, int]: The current epoch and step
                from which you should resume the training
        """

        print("Looking for checkpoints...")
        # look for the checkpoints
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=None,
        )

        # check if a checkpoint exists
        if path is not None:
            print("Loading ...")

            if self.config.deepspeed_enable:
                # try to load the checkpoint
                try:
                    _, client_state = self.model_engine.load_checkpoint(path)
                except Exception:
                    print(
                        "Checkpoint corrupted!"
                        "Try to remove the last checkpoint."
                        "Now Starting from epoch 0, step 0"
                    )
                    return 0, 0
                # load epoch and step to resume loops
                epoch = client_state["epoch"]
                step = client_state["step"]
            else:
                # try to load the checkpoint
                try:
                    checkpoint = torch.load(path)
                except Exception:
                    print(
                        "Checkpoint corrupted!"
                        "Try to remove the last checkpoint."
                        "Now Starting from epoch 0, step 0"
                    )
                    return 0, 0

                # load the model parameters and optimizer parameters
                # from the checkpoint
                epoch = checkpoint["epoch"]
                self.reward.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
                self.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )
                self.training_stats = checkpoint["training_stats"]
                step = checkpoint["step"]
            return epoch, step + 1  # return the next episode to train
        return 0, 0

    def train(
        self,
    ) -> None:
        """Train the reward model"""
        print("Start Training the Reward Model")

        # get config parameters
        if self.config.deepspeed_enable:
            batch_size = self.train_dataloader.batch_size
        else:
            batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device
        iteration_per_print = self.config.iteration_per_print
        checkpoint_steps = self.config.checkpoint_steps

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # load checkpoint
        start_epoch, start_step = self.load_checkpoint()

        # counter for the checkpoint
        cnt_checkpoints = 1

        # traing loop
        for epoch in range(start_epoch, epochs):
            self.reward.train()
            for i, inputs in enumerate(self.train_dataloader):

                # skip the steps if resuming from a checkpoint
                if i < start_step:
                    continue

                # get the inputs
                input_text = inputs[0]
                score = inputs[1]

                # tokenize the input
                with torch.no_grad():
                    input_tokens = self.reward.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )
                    output = torch.as_tensor(
                        score, dtype=torch.float32, device=device
                    )

                # forward pass
                if self.config.deepspeed_enable:
                    est_output = self.model_engine(
                        input_tokens["input_ids"].to(device),
                        input_tokens["attention_mask"].to(device),
                    )[:, -1]
                else:
                    est_output = self.reward.get_reward(
                        input_tokens["input_ids"].to(device),
                        input_tokens["attention_mask"].to(device),
                    )

                # compute the loss
                loss = self.loss_function(est_output, output)
                self.training_stats.training_loss.append(loss.item())

                # backward pass
                if self.config.deepspeed_enable:
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                elif self.config.accelerate_enable:
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                # print progress
                if i % iteration_per_print == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss.item()}"
                    )
                    printed_est_output = [
                        round(float(x), 1) for x in est_output.cpu().tolist()
                    ]
                    print(
                        "prediction",
                        printed_est_output,
                        "target",
                        score.cpu().tolist(),
                    )

                # checkpoints saving
                if cnt_checkpoints % checkpoint_steps == 0:
                    self.save_checkpoint(epoch, i, epochs, n_iter)
                    cnt_checkpoints = 1
                else:
                    cnt_checkpoints += 1

            # Validation
            if self.validation_flag:
                self.reward.eval()
                with torch.no_grad():
                    for i, (text, score) in enumerate(
                        self.validation_dataloader
                    ):

                        # tokenize inputs
                        input_tokens = self.reward.tokenizer(
                            text, return_tensors="pt", padding=True
                        )
                        input_tokens = input_tokens.to(device)
                        # TODO: check on the length of the input tokens if
                        # they are too many it can create problems
                        output = torch.tensor(score, dtype=torch.float32).to(
                            device
                        )

                        # forward pass
                        est_output = self.reward.get_reward(
                            input_tokens["input_ids"],
                            input_tokens["attention_mask"],
                        )

                        # compute loss
                        loss = self.loss_function(est_output, output)
                        self.training_stats.validation_loss.append(loss.item())

                        # print progress
                        if i % iteration_per_print == 0:
                            print(
                                f"Epoch: {epoch+1}/{epochs}, "
                                f"Iteration: {i+1}/{n_iter}, "
                                f"Validation Loss: {loss.item()}"
                            )
            # reset start_step after training is resumed
            start_step = 0

        # save the model at the end of the training
        self.reward.save()
