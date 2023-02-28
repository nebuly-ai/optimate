import json
import os

import deepspeed
import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from config import ConfigActor
from utils import TrainingStats

from chatllama.llama_model import load_model


class ActorModel(torch.nn.Module):
    """Actor model that generates the augmented prompt from the initial
    user_input. The aim is to train this model to generate better prompts.

    Attributes:
        model: The model from LLaMA to be used
        tokenizer: The LLaMA tokenizer
        max_model_tokens (int): Maximum number of tokens that the model can
            handle
        config (ConfigActor): Configuration for the actor model

    Methods:
        load: Load the model from a path
        save: Save the model to a path
        forward: Compute the action logits for a given sequence.
        generate: Generate a sequence from a given prompt
    """

    def __init__(self, config: ConfigActor) -> None:
        super().__init__()
        # load the model

        self.max_model_tokens = 1024
        self.model, self.tokenizer = load_model(
            ckpt_dir=config.model_folder,
            tokenizer_path=config.tokenizer_folder,
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            world_size=int(os.environ.get("WORLD_SIZE", -1)),
            max_batch_size=config.batch_size,
        )
        # save config
        self.config = config

    def parameters(self, **kwargs):
        """Return the parameters of the model

        Args:
            **kwargs:
        """
        return self.model.parameters()

    @beartype
    def forward(
        self, sequences: torch.Tensor, sequences_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate logits to have probability distribution over the vocabulary
            of the actions

        Args:
            sequences (torch.Tensor): Sequences of states and actions used to
                    compute token logits for the whole list of sequences
            attention_mask (torch.Tensor): Mask for the sequences attention

        Returns:
            logits (torch.Tensor): Logits for the actions taken
        """
        model_output = self.model.forward(
            sequences, attention_mask=sequences_mask
        )
        if self.config.debug:
            print("ActorModel.forward")
            print("model_output_logits shape", model_output.logits.shape)
            print("model_output logits", model_output.logits)
        return model_output.logits

    @beartype
    @torch.no_grad()
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions and sequences=[states, actions] from state
            (i.e. input of the prompt generator model)

        Args:
            state (torch.Tensor): the input of the user
            state_mask (torch.Tensor): Mask for the state input (for padding)

        Returns:
            actions (torch.Tensor): Actions generated from the state
            sequences (torch.Tensor): Sequences generated from the
                state as [states, actions]
        """
        max_sequence = states.shape[1]
        max_tokens = self.config.max_tokens + max_sequence
        temperature = self.config.temperature
        # What if the states + completion are longer than the max context of
        # the model?
        sequences = self.model.generate(
            inputs=states,
            attention_mask=state_mask,
            max_length=max_tokens,
            temperature=temperature,
        )
        actions = sequences[:, states.shape[1] :]  # noqa E203
        if self.config.debug:
            print("ActorModel.generate")
            print("state", states)
            print("state shape", states.shape)
            print("sequence shape", sequences.shape)
            print("sequence", sequences)
            print("actions shape", actions.shape)
            print("actions", actions)
        return actions, sequences

    @beartype
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path

        Args:
            path (str): Path to the model
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)
                print(
                    f"Impossible to load the model: {path}"
                    f"The path doesn't exist."
                )
                return
        # load the model
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model: {path}"
                f"The path doesn't exist."
            )
            return
        model_dict = torch.load(path)
        self.model.load_state_dict(model_dict["model"])

    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path

        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)
        torch.save({"model": self.model.state_dict()}, path)


class ActorDataset(Dataset):
    """Dataset for the pretraining of the actor model
    read a json file with the following format:
    [
        {
            "user_input": "..."
            "completion": "..."
        } ,
        ...
    ]
    Where:
        user_input: the input of the user
        completion: the output of the user
    """

    def __init__(self, path: str, device: torch.device) -> None:
        self.device = device
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
            self.data = [
                d["user_input"] + "\n\n###\n\n" + d["completion"] for d in data
            ]
        self.len = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(
        self,
    ):
        return self.len


class ActorTrainer:
    """Used to pre-train the actor model to generate better prompts.

    Args:
        config (ConfigActor): Configuration for the actor model

    Attributes:
        config (ConfigActor): Configuration for the actor model
        model (ActorModel): Actor model
        loss_function (torch.nn.CrossEntropyLoss): Loss function
        optimizer (torch.optim.Adam): Optimizer
        validation_flag (bool): Flag to indicate if the validation dataset
            is provided
        training_stats (TrainingStats): Training statistics

    Methods:
        train: Train the actor model
    """

    def __init__(self, config: ConfigActor) -> None:
        # load the model, optimizer, loss function and config
        self.config = config
        self.model = ActorModel(config)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )

        # check checkpoint, datasets and other data
        if not os.path.exists(config.model_folder):
            os.mkdir(config.model_folder)
        self.validation_flag = False
        self.training_stats = TrainingStats()
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataloaders
        self.train_dataset = ActorDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            self.eval_dataset = ActorDataset(config.validation_dataset_path)
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
        print("Start Actor Model Pretraining")
        # get config parameters
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # traing loop
        for epoch in range(epochs):
            self.model.train()
            for i, input_output in enumerate(self.train_dataloader):
                input_output_tokenized = self.model.tokenizer(
                    input_output,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                training_output = input_output_tokenized["input_ids"][:, 1:]
                training_input = input_output_tokenized["input_ids"][:, :-1]
                attention_mask = input_output_tokenized["attention_mask"][
                    :, :-1
                ]
                training_output = training_output.to(device)
                training_input = training_input.to(device)
                attention_mask = attention_mask.to(device)

                # forward pass
                if self.config.deepspeed_enable:
                    est_output = self.model_engine(
                        training_input, attention_mask
                    )
                else:
                    est_output = self.model.forward(
                        training_input, attention_mask
                    )
                est_output = rearrange(est_output, "b s v -> (b s) v")
                training_output = rearrange(training_output, "b s -> (b s)")
                loss = self.loss_function(est_output, training_output)
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
                if i % self.config.iteration_per_print == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss}"
                    )
            if self.validation_flag:
                self.model.eval()
                for i, input_output in enumerate(self.validation_dataloader):
                    input_output_tokenized = self.model.tokenizer(
                        input_output, return_tensors="pt", padding=True
                    )
                    validation_output = input_output_tokenized["input_ids"][
                        :, 1:
                    ]
                    validation_input = input_output_tokenized["input_ids"][
                        :, :-1
                    ]
                    attention_mask = input_output_tokenized["attention_mask"][
                        :, :-1
                    ]

                    # forward pass
                    est_output = self.model.forward(
                        validation_input, attention_mask
                    )
                    validation_output = rearrange(
                        validation_output, "b s -> (b s)"
                    )
                    est_output = rearrange(est_output, "b s v -> (b s) v")
                    loss = self.loss_function(est_output, validation_output)
                    self.training_stats.validation_loss.append(loss.item())
                    # print progress
                    if i % self.config.iteration_per_print == 0:
                        print(
                            f"Epoch: {epoch+1}/{epochs}, "
                            f"Iteration: {i+1}/{n_iter}, "
                            f"Validation Loss: {loss}"
                        )
        self.model.save()
        print("Training Finished ")
