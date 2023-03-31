import json

import torch
from beartype import beartype
from beartype.typing import Tuple
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

from chatllama.rlhf.base_model import BaseModel, BaseTrainer
from chatllama.rlhf.config import ConfigActor
from chatllama.rlhf.model_list import (
    hf_models_causal_lm,
)
from chatllama.rlhf.dataset import BaseDataset
from chatllama.rlhf.utils import my_logger


class ActorModel(BaseModel):
    """Actor model that generates the augmented prompt from the initial
    user_input. The aim is to train this model to generate better prompts.

    Methods:
        forward: Compute the action logits for a given sequence.
        generate: Generate a sequence from a given prompt
    """

    def __init__(self, config: ConfigActor) -> None:
        super().__init__(config)

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
        # need to return logits for the actions
        if self.config.model in hf_models_causal_lm:
            model_output = model_output.logits
        return model_output

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
        # temperature for the actor
        temperature = self.config.temperature

        # max sequence length for the actor (i.e. prompt + completion)
        max_sequence_length = self.config.max_sequence_length

        # max and min number of tokens to generate
        max_tokens = self.config.max_tokens
        min_tokens = self.config.min_tokens

        # max generation possible given the state and the max sequence length
        max_generation_possible = max_sequence_length - states.shape[1]
        if max_generation_possible < min_tokens:
            raise my_logger.error(
                ValueError,
                f"The prompt is too long w.r.t the "
                f"model sequence length \n"
                f"max_sequence_length={max_sequence_length}\n"
                f"state_length={states.shape[1]}\n"
                f"min_tokens={min_tokens}\n"
                f"max_tokens={max_tokens}\n"
                f"max_generation_possible={max_generation_possible}\n",
            )

        # take the minimum the max_tokens and the max_generation_possible
        max_completion = min(max_tokens, max_generation_possible)

        sequences = self.model.generate(
            input_ids=states,
            attention_mask=state_mask,
            temperature=temperature,
            max_new_tokens=max_completion,
            no_repeat_ngram_size=3,
        )
        actions = sequences[:, states.shape[1] :]  # noqa E203
        return actions, sequences


class ActorDataset(Dataset):
    """Dataset for the pretraining of the actor model
    read a json file with the following format:
    [
        {
            "user_input": "..."
            "completion": "..."
        },
        ...
    ]
    Where:
        user_input: the input of the user
        completion: the output of the user
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
        self.data = [d["user_input"] + d["completion"] for d in data]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(
        self,
    ):
        return len(self.data)


class ActorTrainer(BaseTrainer):
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
        train_dataset (ActorDataset): Training dataset
        train_dataloader (DataLoader): Training dataloader
        validation_dataset (ActorDataset): Validation dataset
        validation_dataloader (DataLoader): Validation dataloader
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        training_stats (TrainingStats): Training statistics
        model_engine (ModelEngine): Model engine for deepspeed training
        accelerator (Accelerator): Accelerator for accelerate training

    Methods:
        train: Train the actor model
    """

    def __init__(self, config: ConfigActor) -> None:

        # store config
        super().__init__(config)

        # load the model
        self.model = ActorModel(config)

        # define loss function
        self.loss_function = torch.nn.CrossEntropyLoss()

        # define optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=1e-5
        )

        # check if validation dataset is provided
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataset and dataloaders
        BaseDataset.clean_dataset(config)
        self.train_dataset = ActorDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            BaseDataset.clean_dataset(config)
            self.eval_dataset = ActorDataset(config.validation_dataset_path)
            self.validation_dataloader = DataLoader(
                self.eval_dataset, batch_size=config.batch_size
            )

        # define scheduler for the learning rate
        # learning rate is decreased until 10% of the initial value
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_dataset) // config.batch_size,
            T_mult=1,
            eta_min=config.lr * 0.1,
        )

        # deepspeed
        self.setup_deepspeed()

        # HF accelerate
        self.setup_accelerate()

        # define the scaler needed for vanilla pytorch with mixed precision
        if (not self.accelerate_enable) and (not self.deepspeed_enable):
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def add_eos_token(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # given tokens and mask, add eos token to the end of each sequence
        # and update the mask
        batch_size, seq_len = tokens.shape
        if self.accelerate_enable:
            eos_token = self.model.module.tokenizer.eos_token_id
        else:
            eos_token = self.model.tokenizer.eos_token_id

        # see if i can append 1 token
        n_tokens_to_append = min(self.config.max_sequence_length - seq_len, 1)
        n_tokens_to_append = max(n_tokens_to_append, 0)

        # concatenate eos to tokens and mask
        if n_tokens_to_append > 0:
            tokens = torch.cat(
                [
                    tokens,
                    torch.ones(batch_size, n_tokens_to_append)
                    .long()
                    .to(tokens.device)
                    * eos_token,
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.ones(batch_size, n_tokens_to_append)
                    .long()
                    .to(mask.device),
                ],
                dim=1,
            )
        return tokens, mask

    def train(
        self,
    ) -> None:
        """Train the model"""

        my_logger.success("Start Actor Model Pretraining")

        # get config parameters
        if self.deepspeed_enable:
            # get batch size from deepspeed
            batch_size = self.model_engine.train_batch_size()
        elif self.accelerate_enable:
            batch_size = (
                self.config.batch_size * self.accelerator.num_processes
            )
        else:
            batch_size = self.config.batch_size

        epochs = self.config.epochs
        device = self.config.device
        checkpoint_steps = self.config.checkpoint_steps

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # load model_checkpoint
        start_epoch, start_step = self.load_checkpoint()

        if start_epoch == 0 and start_step == 0:
            self.training_stats.clear()

        # counter for the checkpoint
        cnt_checkpoint = 1

        # traing loop
        for epoch in range(start_epoch, epochs):
            self.model.train()
            for i, input_text in enumerate(self.train_dataloader):

                # skip the first steps if we are resuming training
                if i < start_step:
                    continue

                # tokenize input
                with torch.no_grad():
                    if self.accelerate_enable:
                        input_tokenized = self.model.module.tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                        )
                    else:
                        input_tokenized = self.model.tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=self.config.max_sequence_length,
                        )

                    # split tokens and mask
                    input_tokenized_id = input_tokenized["input_ids"]
                    input_tokenized_mask = input_tokenized["attention_mask"]

                    # add eos token
                    (
                        input_tokenized_id,
                        input_tokenized_mask,
                    ) = self.add_eos_token(
                        input_tokenized_id,
                        input_tokenized_mask,
                    )

                    # split into input and output
                    training_output = input_tokenized_id[:, 1:]
                    training_input = input_tokenized_id[:, :-1]
                    attention_mask = input_tokenized_mask[:, :-1]

                    # move to device
                    training_output = training_output.to(device)
                    training_input = training_input.to(device)
                    attention_mask = attention_mask.to(device)

                # forward pass
                if self.deepspeed_enable:
                    est_output = self.model_engine(
                        training_input, attention_mask
                    )
                elif self.accelerate_enable:
                    est_output = self.model(training_input, attention_mask)
                else:
                    with torch.autocast(
                        device_type=self.config.device_type,
                        dtype=torch.float16,
                    ):
                        est_output = self.model(training_input, attention_mask)

                # compute loss
                if (not self.accelerate_enable) and (
                    not self.deepspeed_enable
                ):

                    # vanilla pytorch use autocast
                    with torch.autocast(
                        device_type=self.config.device_type,
                        dtype=torch.float16,
                    ):
                        est_output = rearrange(est_output, "b s v -> (b s) v")
                        training_output = rearrange(
                            training_output, "b s -> (b s)"
                        )
                        loss = self.loss_function(est_output, training_output)
                else:

                    # deepspeed and accelerate use defualt
                    est_output = rearrange(est_output, "b s v -> (b s) v")
                    training_output = rearrange(
                        training_output, "b s -> (b s)"
                    )
                    loss = self.loss_function(est_output, training_output)

                # save training stats
                self.append_training_stats(training_loss=loss.item())

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
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                # print progress
                if i % self.config.iteration_per_print == 0:
                    my_logger.info(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss}"
                    )

                # save checkpoint periodically
                if cnt_checkpoint % checkpoint_steps == 0:
                    self.save_checkpoint(epoch, i, epochs, n_iter)
                    self.training_stats.save()
                    cnt_checkpoint = 1
                else:
                    cnt_checkpoint += 1

            # Validation
            if self.validation_flag:
                self.model.eval()
                with torch.no_grad():
                    for i, input_text in enumerate(self.validation_dataloader):

                        # tokenize input
                        input_tokenized = self.model.tokenizer(
                            input_text, return_tensors="pt", padding=True
                        )
                        validation_output = input_tokenized["input_ids"][:, 1:]
                        validation_input = input_tokenized["input_ids"][:, :-1]
                        attention_mask = input_tokenized["attention_mask"][
                            :, :-1
                        ]

                        # forward pass
                        est_output = self.model.forward(
                            validation_input, attention_mask
                        )
                        validation_output = rearrange(
                            validation_output, "b s -> (b s)"
                        )

                        # compute loss
                        est_output = rearrange(est_output, "b s v -> (b s) v")
                        loss = self.loss_function(
                            est_output, validation_output
                        )
                        self.append_training_stats(validation_loss=loss.item())

                        # print progress
                        if i % self.config.iteration_per_print == 0:
                            my_logger.info(
                                f"Epoch: {epoch+1}/{epochs}, "
                                f"Iteration: {i+1}/{n_iter}, "
                                f"Validation Loss: {loss}"
                            )
            # reset start_step after training is resumed
            start_step = 0

        # save the model
        if self.accelerate_enable:
            self.model.module.save()
        else:
            self.model.save()
        my_logger.success("Training Finished ")
