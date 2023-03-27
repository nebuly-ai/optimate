import json
import yaml
import os
import shutil

import deepspeed
import torch
from accelerate import Accelerator
from beartype import beartype
from beartype.typing import Tuple
from einops import rearrange
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from chatllama.rlhf.config import ConfigActor
from chatllama.rlhf.model_list import (
    hf_models_causal_lm,
    llama_models,
    hf_models,
)

from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.utils import TrainingStats


class ActorModel(torch.nn.Module):
    """Actor model that generates the augmented prompt from the initial
    user_input. The aim is to train this model to generate better prompts.

    Attributes:
        model: The model from LLaMA to be used
        tokenizer: The LLaMA tokenizer
        config (ConfigActor): Configuration for the actor model

    Methods:
        load: Load the model from a path
        save: Save the model to a path
        forward: Compute the action logits for a given sequence.
        generate: Generate a sequence from a given prompt
    """

    def __init__(self, config: ConfigActor) -> None:
        super().__init__()

        # save config
        self.config = config

        # initialize the self.model
        if config.model in llama_models:
            # llama module might not be present when HF models are used
            from chatllama.llama_model import (
                load_model,
                setup_model_parallel,
            )  # noqa

            local_rank, world_size = setup_model_parallel()

            # use load_model_test for testing
            self.model, self.tokenizer = load_model(
                ckpt_dir=config.model_folder,
                tokenizer_path=config.tokenizer_path,
                local_rank=local_rank,
                world_size=world_size,
                froze_embeddings=config.froze_embeddings,
                use_fairscale=config.use_fairscale,
                max_batch_size=config.batch_size,
            )
        elif config.model in hf_models_causal_lm:
            self.tokenizer = self.load_tokenizer(config)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
            )

            # Setup PEFT model
            if config.peft_enable:

                # check that the peft config exist
                if os.path.exists(config.peft_config_path):
                    # Read the peft config from yaml
                    with open(config.peft_config_path, "r") as c:
                        config_peft = yaml.safe_load(c)
                else:
                    raise ValueError(
                        f"PEFT config {config.peft_config_path} not found"
                    )

                print(config_peft)
                # define lora config for peft
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, **config_peft
                )

                # create peft model
                self.model = get_peft_model(
                    model=self.model,
                    peft_config=peft_config,
                )

            self.model.to(config.device)

        else:
            raise ValueError(f"Model {config.model} not supported")

        # load the model from model_folder
        self.load()

    @beartype
    def load(self) -> None:
        """Load the model from the path"""
        # check if there is a model to load
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=False,
            current_epoch=None,
        )

        # if there is a model to load
        if path is not None:

            # load the model
            print("Loading ...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict["state_dict"])

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
            {"state_dict": self.model.state_dict()},
            path,
        )

    @staticmethod
    def load_tokenizer(config: ConfigActor):
        """Load the tokenizer from the model name"""
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

            # llama module might not be present when HF models are used
            from chatllama.llama_model import (
                load_tokenizer,
            )  # noqa

            tokenizer = load_tokenizer(config.tokenizer_path)
        return tokenizer

    def parameters(self):
        """Return the parameters of the model"""
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
        # need to return logits for the actions
        if self.config.model in hf_models_causal_lm:
            model_output = model_output.logits
        if self.config.debug:
            print("ActorModel.forward")
            print("model_output_logits shape", model_output.shape)
            print("model_output logits", model_output)
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
            raise ValueError(
                f"The prompt is too long w.r.t the "
                f"model sequence length \n"
                f"max_sequence_length={max_sequence_length}\n"
                f"state_length={states.shape[1]}\n"
                f"min_tokens={min_tokens}\n"
                f"max_tokens={max_tokens}\n"
                f"max_generation_possible={max_generation_possible}\n"
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
        if self.config.debug:
            print(
                f"input length {states.shape[1]} \n"
                f"max sequence length {max_sequence_length} \n"
                f"max completion {max_completion} \n"
                f"generated sequence {sequences.shape[1]} \n"
            )
            print("ActorModel.generate")
            print("state", states)
            print("state shape", states.shape)
            print("sequence shape", sequences.shape)
            print("sequence", sequences)
            print("actions shape", actions.shape)
            print("actions", actions)
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
        load_checkpoint: Load a checkpoint
        save_checkpoint: Save a checkpoint
    """

    def __init__(self, config: ConfigActor) -> None:

        # store config
        self.config = config

        # load the model
        self.actor = ActorModel(config)

        # define loss function
        self.loss_function = torch.nn.CrossEntropyLoss()

        # define optimizer
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.lr, weight_decay=1e-5
        )

        # check if validation dataset is provided
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataset and dataloaders
        self.train_dataset = ActorDataset(config.train_dataset_path)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
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

        # define training statistics
        stat_path = ModelLoader.get_training_stats_path(config)
        self.training_stats = TrainingStats(stat_path)

        # consistency check between accelerate and deepspeed
        if config.accelerate_enable and config.deepspeed_enable:
            raise ValueError(
                "Both DeepSpeed and Accelerate are enabled for the Actor."
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
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.actor,
                model_parameters=self.actor.parameters(),
                training_data=self.train_dataset,
                config=self.config.deepspeed_config_path,
            )
            print("Training with DeepSpeed")

        # initialize accelerate
        self.accelerator = None
        if config.accelerate_enable is True:
            self.accelerator = Accelerator()
            (
                self.actor,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.actor,
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
        """Save the current checkpoint

        Args:
            current_epoch (int): Current epoch
            current_step (int): Current step
            max_epochs (int): Maximum number of epochs
            max_steps (int): Maximum number of steps
        """

        print(
            f"Saving checkpoint for epoch {current_epoch + 1}, "
            f"step {current_step + 1} ..."
        )
        # look for path to save the checkpoint
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

        if self.config.deepspeed_enable:
            client_state = {
                "epoch": current_epoch,
                "step": current_step,
            }
            self.model_engine.save_checkpoint(path, client_state=client_state)
        else:
            # save the checkpoint
            torch.save(
                {
                    "state_dict": self.actor.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "training_stats": self.training_stats,
                    "epoch": current_epoch,
                    "step": current_step,
                },
                path,
            )

        # remove old checkpoints
        n_checkpoints_to_keep = self.config.n_checkpoints_to_keep
        ModelLoader.delete_old_checkpoints(
            model_folder, model_name, n_checkpoints_to_keep
        )

    @beartype
    def load_checkpoint(
        self,
    ) -> Tuple[int, int]:
        """Load a checkpoint from the model folder

        Returns:
            Tuple[int, int]: Current epoch and current step to resume
                training
        """

        print("Looking for checkpoints...")
        # look for a checkpoint
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=None,
        )

        # if there is a checkpoint
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

                # assing the checkpoint to the model
                epoch = checkpoint["epoch"]
                self.actor.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
                self.trainign_stats = checkpoint["training_stats"]
                step = checkpoint["step"]
                return epoch, step + 1  # return the next episode to train
        return 0, 0

    def add_eos_token(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # given tokens and mask, add eos token to the end of each sequence
        # and update the mask
        batch_size, seq_len = tokens.shape
        eos_token = self.actor.tokenizer.eos_token_id

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
        print("Start Actor Model Pretraining")

        # get config parameters
        if self.config.deepspeed_enable:
            batch_size = self.train_dataloader.batch_size
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
            self.actor.train()
            for i, input_text in enumerate(self.train_dataloader):

                # skip the first steps if we are resuming training
                if i < start_step:
                    continue

                # tokenize input
                with torch.no_grad():
                    input_tokenized = self.actor.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
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
                if self.config.deepspeed_enable:
                    est_output = self.model_engine(
                        training_input, attention_mask
                    )
                else:
                    est_output = self.actor(training_input, attention_mask)

                # compute loss
                est_output = rearrange(est_output, "b s v -> (b s) v")
                training_output = rearrange(training_output, "b s -> (b s)")
                loss = self.loss_function(est_output, training_output)
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
                if i % self.config.iteration_per_print == 0:
                    print(
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
                self.actor.eval()
                with torch.no_grad():
                    for i, input_text in enumerate(self.validation_dataloader):

                        # tokenize input
                        input_tokenized = self.actor.tokenizer(
                            input_text, return_tensors="pt", padding=True
                        )
                        validation_output = input_tokenized["input_ids"][:, 1:]
                        validation_input = input_tokenized["input_ids"][:, :-1]
                        attention_mask = input_tokenized["attention_mask"][
                            :, :-1
                        ]

                        # forward pass
                        est_output = self.actor.forward(
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
                        self.training_stats.validation_loss.append(loss.item())

                        # print progress
                        if i % self.config.iteration_per_print == 0:
                            print(
                                f"Epoch: {epoch+1}/{epochs}, "
                                f"Iteration: {i+1}/{n_iter}, "
                                f"Validation Loss: {loss}"
                            )
            # reset start_step after training is resumed
            start_step = 0

        # save the model
        self.actor.save()
        print("Training Finished ")
