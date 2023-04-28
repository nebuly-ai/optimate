import json
import os
import random
from collections import deque, namedtuple

import torch
from beartype import beartype
from beartype.typing import Deque, List, Tuple, Union
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset

from chatllama.rlhf.actor import ActorModel
from chatllama.rlhf.base_model import BaseModel, BaseTrainer
from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)
from chatllama.rlhf.model_list import hf_models
from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.reward import RewardModel, CriticModel
from chatllama.rlhf.utils import ConversationLog, my_logger


"""
train()
┌─────────────────────────────┐
│                             │◄─────────────────────────┐
│                             │                          │
│      ┌─────────────┐        │                          │
│      │ user input  │        │                          │ learn()
│      └─────┬───────┘        │             ┌────────────┴─────────────┐
│            │                │             │                          │
│            │                │             │       ┌────────┐         │
│            │                │             │   ┌───│ Update │──┐      │
│            │                │             │   │   └────▲───┘  │      │
│   ┌────────▼────────────┐   │             │   │        │      │      │
│   │  Actor (LLM Model)  │   │             │   │     ┌──┴───┐  │      │
│   └────────┬────────────┘   │             │   │     │ PPO  │  │      │
│            │                │             │   │     └▲────▲┘  │      │
│            │                │             │   │      │    │   │      │
│            │                │             │   │      │    │   │      │
│    ┌───────▼──────┐         │             │ ┌─▼──────┴┐ ┌─┴───▼──┐   │
│    │ Reward Model │         │             │ │  Actor  │ │ Critic │   │
│    └──────────────┘         │             │ └─────────┘ └────────┘   │
│                             │             │                          │
│                             │ x Episodes  └─────────────▲────────────┘
└───────────────┬─────────────┘                           │   x Epochs
                │ store N Examples per Timestep           │  
         ┌──────▼──────┐                                  │
         │             │                                  │
         │  Memories   ├──────────────────────────────────┘
         │             │ (update timesteps x N Examples)
         └─────────────┘
"""  # noqa W291


def change_tokenization(tokens, tokenizer1, tokenizer2):
    """Change the tokenizer of the tokens

    Args:
        tokens (torch.Tensor): Tokens to be changed
        tokenizer1 (transformers.PreTrainedTokenizer): Tokenizer to be changed
        tokenizer2 (transformers.PreTrainedTokenizer): Tokenizer to be
            changed to

    Returns:
        encoded_tokens: Encoded tokens
    """

    # decode tokens
    with torch.no_grad():
        decoded_tokens = [
            tokenizer1.decode(token) for i, token in enumerate(tokens)
        ]

        # remove all the pad tokens
        decoded_tokens = [
            token.replace(tokenizer1.pad_token, "") for token in decoded_tokens
        ]

        # remove all the eos tokens
        decoded_tokens = [
            token.replace(tokenizer1.eos_token, "") for token in decoded_tokens
        ]

        # encode the actions with critic tokenizer
        encoded_tokens = tokenizer2(
            decoded_tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    return encoded_tokens


ConfigType = Union[ConfigActor, ConfigReward, ConfigCritic]


@beartype
def check_model_family(config1: ConfigType, config2: ConfigType) -> bool:
    """Check if the model family is the same for the two configs
    the model family is specified in the config.model

    Args:
        config1 (ConfigType): First config
        config2 (ConfigType): Second config

    Returns:
        bool: True if the model family is the same, False otherwise
    """

    # check if both are an hugging face models
    if (config1.model in hf_models) and (config2.model in hf_models):

        # if there is a "/" remove it from the name
        model_name1 = config1.model
        model_name2 = config2.model
        if "/" in model_name1:
            model_name1 = model_name1.split("/")[1]
        if "/" in model_name2:
            model_name2 = model_name2.split("/")[1]

        # check if the model family is the same
        return model_name1.split("-")[0] == model_name2.split("-")[0]

    # check if both are not an hugging face models
    elif (config1.model not in hf_models) and (config2.model not in hf_models):

        # for now they could be only LLaMA models
        return True
    else:
        return False


class ActorCritic(BaseModel):
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model
        use_same_tokenizer (bool): True if actor and critic use the same
            tokenizer, False otherwise

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # load the actor
        self.actor = ActorModel(config.actor)

        # check if critic must be initialized from reward model
        ModelLoader.init_critic_from_reward(config.critic)

        # now load the critic
        self.critic = CriticModel(config.critic)

        # flag to check if actor and critic use the same tokenizer
        self.use_same_tokenizer = check_model_family(
            config.actor, config.critic
        )

    @beartype
    def forward(
        self,
        sequences_actor: torch.Tensor,
        sequences_mask_actor: torch.Tensor,
        sequences_critic: torch.Tensor,
        sequences_mask_critic: torch.Tensor,
        action_len_actor: int,
        action_len_critic: int,
    ) -> Tuple:
        """Given the whole sequences, use the actor forward to get the logits
            for each token in the sequence and the critic forward to get the
            values for each generation step.

        Args:
            sequences_actor (torch.Tensor) [batch_size, seq_len
                composed of [states, actions] for the actor
            sequence_mask_actor (torch.Tensor) [batch_size, seq_len]: Mask
                for the sequences of the actor
            sequences_critic (torch.Tensor) [batch_size, seq_len]: Sequences
                composed of [states, actions] for the critic.
                Can differ from sequences_actor if the critic uses a different
                tokenizer
            sequences_mask_critic (torch.Tensor) [batch_size, seq_len]: Mask
                for the sequences of the critic.
            action_len_actor (int): Length of the actions in the sequences
                for the actor
            action_len_critic (int): Length of the actions in the sequences
                for the critic

        Returns:
            action_logits (torch.Tensor) [batch_size, seq_len, vocab_size]:
                Logits for the actions in the sequences
            values (torch.Tensor) [batch_size, seq_len]: Values for the actions
                in the sequences
        """

        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        actions_logits = self.actor.forward(
            sequences_actor, sequences_mask_actor
        )

        # use the critic forward to get the values for the actions
        values = self.critic.forward(sequences_critic, sequences_mask_critic)

        # return only logits and values for the actions taken
        real_actions_logits = actions_logits[:, -action_len_actor:, :]
        real_values = values[:, -action_len_critic:]

        return (
            real_actions_logits,
            real_values,
        )

    @torch.no_grad()
    @beartype
    def generate(
        self,
        states_actor: torch.Tensor,
        states_mask_actor: torch.Tensor,
        states_critic: torch.Tensor,
    ) -> Tuple:
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states_actor (torch.Tensor) [batch_size, input_len]: States for the
                actor.
            states_mask_actor (torch.Tensor) [batch_size, input_len]: Mask for
                the states for the actor
            states_critic (torch.Tensor) [batch_size, input_len]: States for
                the critic. Can differ from states_actor if the critic uses a
                different tokenizer

        Returns:
            actions (torch.Tensor) [batch_size, act_len]: Actions generated
                from the states.
            actions_logits (torch.Tensor) [batch_size, act_len, vocab_size]:
                Logits for the actions generated from the states
                (i.e. pi(y | x))
            values (torch.Tensor) [batch_size, act_len]: Values generated by
                the critic model for the actions generated by the actor
                (i.e. V(x))
            sequences (torch.Tensor) [batch_size, seq_len]: Sequences
                generated from the states as [states, actions]
        """

        # generate action sequence from the actor
        actions, sequences_actor = self.actor.generate(
            states_actor, states_mask_actor
        )

        # create mask for the actor sequences
        sequences_mask_actor = (
            (sequences_actor != self.actor.tokenizer.pad_token_id)
            .to(sequences_actor.device)
            .long()
            .detach()
        )

        # get the length of the actions
        action_len_actor = actions.shape[1]

        # check if different encoding is needed for the critic
        if self.use_same_tokenizer:
            sequences_critic = sequences_actor
            sequences_mask_critic = sequences_mask_actor
            action_len_critic = action_len_actor
        else:
            encoded_critic = change_tokenization(
                sequences_actor,
                self.actor.tokenizer,
                self.critic.tokenizer,
            )
            # split the encoded_critic in tokens and maks
            sequences_critic = encoded_critic["input_ids"].to(
                sequences_actor.device,
            )
            sequences_mask_critic = (
                encoded_critic["attention_mask"]
                .to(sequences_actor.device)
                .long()
                .detach()
            )

            # compute len of actions for the critic tokenizer
            action_len_critic = states_critic.shape[1]

        # generate actions_logits and values
        actions_logits, values = self.forward(
            sequences_actor,
            sequences_mask_actor,
            sequences_critic,
            sequences_mask_critic,
            action_len_actor,
            action_len_critic,
        )

        return (
            actions,
            actions_logits,
            values,
            sequences_actor,
            sequences_mask_actor,
            sequences_critic,
            sequences_mask_critic,
            action_len_actor,
            action_len_critic,
        )


# structure to store the data for each experience
Memory = namedtuple(
    "Memory",
    [
        "values",
        "rewards",
        "actions_log_probs",
        "sequences_actor",
        "sequences_mask_actor",
        "sequences_critic",
        "sequences_mask_critic",
        "action_len_actor",
        "action_len_critic",
    ],
)


class ExperienceDataset(Dataset):
    """Dataset to train the actor-critic models"""

    def __init__(
        self,
        memories: Deque[Memory],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.data = list(memories)

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].values,
            self.data[idx].rewards,
            self.data[idx].actions_log_probs,
            self.data[idx].sequences_actor,
            self.data[idx].sequences_mask_actor,
            self.data[idx].sequences_critic,
            self.data[idx].sequences_mask_critic,
            int(self.data[idx].action_len_actor),
            int(self.data[idx].action_len_critic),
        )
        return item


class ExamplesSampler:
    """Store the prompt to be sampled to generate the examples
    read a json file with the following format:
    [
        {
            "user_input" : "",
        } ,
        ...
    ]
    Where:
        user_input: is the input of the user or directly the input of the user
            with the memory preappended (i.e. user_input + memory)
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
        self.data = [d["user_input"] for d in data]

    def sample(self, n: int) -> List:
        """Sample n examples from the data

        Args:
            n (int): Number of examples to sample
        """
        return random.sample(self.data, n)


class CustomOptimizer(torch.optim.Optimizer):
    """Class to define two different LR for the actor and the critic.
    Still not working with distributed trainig.
    """

    def __init__(self, params_actor, params_critic, lr_actor, lr_critic):
        self.actor = torch.optim.AdamW(params_actor, lr=lr_actor)
        self.critic = torch.optim.AdamW(params_critic, lr=lr_critic)

        self.param_groups = self.actor.param_groups + self.critic.param_groups

    def step(self):
        self.actor.step()
        self.critic.step()

    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()


class RLTrainer(BaseTrainer):
    """Train the actor-critic model using RL

    Attributes:
        config (Config): Configuration of the trainer
        actorcritic (ActorCritic): Actor-critic model
        reward (Reward): Reward function
        optimizer (torch.optim.Optimizer): Optimizer for the actor-critic model
        scheduler (torch.optim.lr_scheduler): Scheduler for the optimizer
        examples_sampler (ExamplesSampler): Sampler to generate the examples
        conversation_log (ConversationLog): List of the conversation logs

    Methods:
        train: the training loop that calls the learn function after generating
            the experiences.
        learn: Learn from a batch of experiences and update the actor and the
            critic model.

    Known Issues:
        - When using load_8bit and peft on the actor model, the following error
            is raised:
            return torch.layer_norm(input,
                        normalized_shape,
                        weight,
                        bias,
                        eps,
                        torch.backends.cudnn.enabled)
            RuntimeError: Expected all tensors to be on the same device,
                but found at least two devices, cuda:1 and cuda:0!
                (when checking argument for argument weight
                in method wrapper_CUDA__native_layer_norm)
    """

    def __init__(
        self,
        config: Config,
    ) -> None:

        # initialize actor-critic
        self.actorcritic = ActorCritic(config)

        # TODO: currently now working with custom optimizer
        # initialize actor optimizer
        # self.optimizer = CustomOptimizer(
        #     self.actorcritic.actor.parameters(),
        #     self.actorcritic.critic.parameters(),
        #     config.trainer.actor_lr,
        #     config.trainer.critic_lr,
        # )

        self.optimizer = torch.optim.AdamW(
            self.actorcritic.parameters(),
            lr=config.trainer.actor_lr,
        )

        # scheduler init to None (need the dataset info to initialize it)
        self.scheduler = None

        # initialize reward model
        self.reward = RewardModel(config.reward)

        # initialize class to store conversations logs
        # initialize class to store conversations logs
        model_folder, _, _ = ModelLoader.get_model_path(
            config,
            is_checkpoint=True,
        )
        path = os.path.join(model_folder, "conversations_log.json")
        self.conversation_log = ConversationLog(path)

        # initialize examples sampler
        self.example_sampler = ExamplesSampler(config.trainer.examples_path)

        # Base Trainer Initialization:
        super().__init__(config)

    def ppo_loss(
        self,
        actions_logits: torch.Tensor,
        old_actions_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the PPO loss

        Args:
            actions_logits (torch.Tensor): Logits of the actions
            old_actions_log_probs (torch.Tensor): Log prob of the actions
            old_values (torch.Tensor): Values of the actions
            values (torch.Tensor): Values of the actions
            rewards (torch.Tensor): Rewards of the actions

        Returns:
            torch.Tensor: PPO loss

        """

        # get hyperparameters
        actor_eps_clip = self.config.trainer.actor_eps_clip
        critic_eps_clip = self.config.trainer.critic_eps_clip
        beta_s = self.config.trainer.beta_s

        # get action log prob
        actions_prob = torch.softmax(actions_logits, dim=-1).max(dim=-1).values
        actions_log_prob = torch.log(actions_prob + self.eps)

        # compute entropy
        entropies = (actions_prob * actions_log_prob).sum(dim=-1)

        # compute KL divergence
        kl_div_loss = (
            (actions_prob * (actions_log_prob - old_actions_log_probs))
            .sum(dim=-1)
            .mean()
        )

        # compute ratios
        ratios = (actions_log_prob - old_actions_log_probs).exp()

        # compute policy loss
        if check_model_family(self.config.actor, self.config.critic):

            # compute discounted rewards as in TRL
            gamma = self.config.trainer.gamma_discounted
            discounted_rewards = torch.zeros_like(old_values)
            gamma_power = torch.arange(
                0,
                discounted_rewards.shape[1],
                device=rewards.device,
                dtype=rewards.dtype,
            )
            gamma_vect = gamma**gamma_power
            for i in range(discounted_rewards.shape[1]):
                discounted_rewards[:, i] = torch.matmul(
                    rewards[:, i:], gamma_vect[: rewards[:, i:].shape[1]]
                )
            advantages = (
                discounted_rewards - old_values
            )  # TRL has opposite sign for old values
            if advantages.std() > self.eps:
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps
                )
            surr1 = advantages * ratios
        else:
            advantages = rewards - old_values[:, -1]
            surr1 = advantages * ratios

        surr2 = (
            torch.clamp(ratios, 1 - actor_eps_clip, 1 + actor_eps_clip)
            * advantages
        )

        policy_loss = -torch.min(surr1, surr2) - beta_s * entropies
        policy_loss = policy_loss.mean()
        policy_loss = policy_loss + kl_div_loss

        # compute value loss
        value_loss_clipped = old_values + (values - old_values).clamp(
            -critic_eps_clip, critic_eps_clip
        )
        value_loss1 = (value_loss_clipped - rewards) ** 2
        value_loss2 = (values - rewards) ** 2
        value_loss = torch.max(value_loss1, value_loss2).mean()

        # Sum the two losses
        loss = policy_loss + value_loss

        # check the losses
        if torch.isnan(loss):
            if torch.isnan(policy_loss):
                if torch.isnan(kl_div_loss):
                    raise my_logger.error(
                        ValueError,
                        "KL div Loss is nan",
                    )
                else:
                    if torch.isnan(entropies.mean()):
                        raise my_logger.error(
                            ValueError,
                            "Entropies Loss is nan",
                        )
                    elif torch.isnan(surr1.mean()) or torch.isnan(
                        surr2.mean()
                    ):
                        if torch.isnan(advantages.mean()):
                            if torch.isnan(rewards.mean()):
                                raise my_logger.error(
                                    ValueError,
                                    "Rewards are nan",
                                )
                            elif torch.isnan(old_values.mean()):
                                raise my_logger.error(
                                    ValueError,
                                    "Old Values are nan",
                                )
                            elif torch.isnan(gamma_vect.mean()):
                                raise my_logger.error(
                                    ValueError,
                                    "Gamma Vector is nan",
                                )
                            else:
                                raise my_logger.error(
                                    ValueError,
                                    "Advantages are nan",
                                )
                        elif torch.isnan(ratios.mean()):
                            raise my_logger.error(
                                ValueError,
                                "Ratios are nan",
                            )
            else:
                raise my_logger.error(
                    ValueError,
                    "Value Loss is nan",
                )
        return loss, policy_loss, value_loss

    @beartype
    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        my_logger.info("Start to Learn...")

        # get parameters
        epochs = self.config.trainer.epochs

        # TODO: check this with distributed training
        batch_size = self.config.trainer.batch_size

        # create dataset from memories
        self.train_dataset = ExperienceDataset(memories, self.device)
        self.train_dataloader = self.create_dataloader(
            self.train_dataset, batch_size
        )

        # assign a scheduler to the optimizer
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_dataset) // batch_size,
            T_mult=1,
            eta_min=self.config.trainer.actor_lr * 0.1,
        )

        # train agent-critic
        self.actorcritic.train()
        for epoch in range(epochs):
            for k, batch in enumerate(self.train_dataloader):

                # TODO: now first two elements are not used can be removed?
                (
                    old_values,
                    rewards,
                    old_actions_log_probs,
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                ) = [tensor.to(self.device) for tensor in batch]

                # get actor critic new probabilities and values
                if self.deepspeed_enable:
                    actions_logits, values = self.model_engine.forward(
                        sequences_actor,
                        sequences_mask_actor,
                        sequences_critic,
                        sequences_mask_critic,
                        action_len_actor.item(),
                        action_len_critic.item(),
                    )
                elif self.accelerate_enable:
                    actions_logits, values = self.actorcritic.forward(
                        sequences_actor,
                        sequences_mask_actor,
                        sequences_critic,
                        sequences_mask_critic,
                        action_len_actor.item(),
                        action_len_critic.item(),
                    )
                else:
                    with torch.autocast(
                        device_type=self.config.trainer.device_type,
                        dtype=torch.float16,
                    ):
                        actions_logits, values = self.actorcritic.forward(
                            sequences_actor,
                            sequences_mask_actor,
                            sequences_critic,
                            sequences_mask_critic,
                            action_len_actor.item(),
                            action_len_critic.item(),
                        )
                # compute the loss
                if self.accelerate_enable or self.deepspeed_enable:
                    (loss, policy_loss, value_loss,) = self.ppo_loss(
                        actions_logits,
                        old_actions_log_probs,
                        old_values,
                        values,
                        rewards,
                    )
                else:
                    with torch.autocast(
                        device_type=self.config.trainer.device_type,
                        dtype=torch.float16,
                    ):
                        (loss, policy_loss, value_loss,) = self.ppo_loss(
                            actions_logits,
                            old_actions_log_probs,
                            old_values,
                            values,
                            rewards,
                        )

                # backward pass
                if self.deepspeed_enable:
                    # DeepSpeed backward pass
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                elif self.accelerate_enable:
                    # Accelerate backward pass
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    # PyTorch mixed precision backward pass
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                # append the losses to the training stats
                self.append_training_stats(
                    training_loss=policy_loss.detach().cpu().item(),
                    value_loss=value_loss.detach().cpu().item(),
                )

                # print iteration info
                my_logger.info(
                    f"Epoch {epoch+1}/{epochs} "
                    f"Step "
                    f"{k+1}/{int(len(self.train_dataloader) / batch_size)} "
                    f"Policy Loss {policy_loss.detach().cpu().item():.4f} "
                    f"Value Loss {value_loss.detach().cpu().item():.4f} "
                )

        self.actorcritic.eval()
        my_logger.info("End Learning")

    def train(
        self,
    ) -> None:

        my_logger.success("Start RL Training")

        # initialize settings
        num_episodes = self.config.trainer.num_episodes
        max_timesteps = self.config.trainer.max_timesteps
        num_examples = self.config.trainer.num_examples
        update_timesteps = self.config.trainer.update_timesteps
        batch_size = self.config.trainer.batch_size
        checkpoint_steps = self.config.trainer.checkpoint_steps

        # number of elements that the memories should contain when learning
        number_of_memories_per_learn_iteration = (
            num_examples * update_timesteps
        )

        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % batch_size == 0
        ), "The number of memories must be a multiple of the batch size"

        # the total number of timesteps done in the train() are
        total_number_of_timesteps = num_episodes * max_timesteps

        # the total timesteps done should be a multiple of the update timesteps
        assert total_number_of_timesteps % update_timesteps == 0, (
            "The number of timesteps (num_episodes*max_timesteps)"
            "must be a multiple of the update_timesteps"
        )

        # initialize memories
        memories = deque([])

        # load checkpoint
        start_episode, _ = self.load_checkpoint()

        # if it is a new training from the start clear the conversation log
        if start_episode == 0:
            self.conversation_log.clear()

        # initialize counters
        cnt_timesteps = 0
        cnt_learn_iter = 0

        # move reward model to correct device
        self.reward.to(self.device)

        # loop over episodes and timesteps
        self.actorcritic.eval()
        for episode in range(start_episode, num_episodes):
            for timestep in range(max_timesteps):

                # ensure that no gradients are computed during this step
                with torch.no_grad():

                    # print the iteration info
                    my_logger.info(
                        f"Episode: {episode + 1}/{num_episodes}, "
                        f"Timestep: {timestep + 1}/{max_timesteps}, "
                        f"Learning Cnt: "
                        f"{cnt_timesteps + 1}/{update_timesteps}"
                    )

                    # counter used to count timesteps into memory
                    cnt_timesteps += 1

                    # sample num_examples examples from  example dataset
                    inputs = self.example_sampler.sample(num_examples)

                    # tokenize examples for the actor
                    if self.accelerate_enable:
                        tok_inputs_act = self.actorcritic.module.actor.tokenizer(  # noqa 501
                            inputs,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                        )
                    else:
                        tok_inputs_act = self.actorcritic.actor.tokenizer(
                            inputs,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                        )

                    states_actor = tok_inputs_act["input_ids"]
                    states_mask_actor = tok_inputs_act["attention_mask"]

                    # tokenize examples for the critic
                    if self.accelerate_enable:
                        tok_inputs_crt = self.actorcritic.module.critic.tokenizer(  # noqa 501
                            inputs,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                        )
                    else:
                        tok_inputs_crt = self.actorcritic.critic.tokenizer(
                            inputs,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                        )

                    states_critic = tok_inputs_crt["input_ids"]

                    # move to device
                    if not self.config.critic.load_8bit:
                        states_critic = states_critic.to(self.device)
                        states_actor = states_actor.to(self.device)
                        states_mask_actor = states_mask_actor.to(self.device)

                    # generate sequences of actions and values
                    if self.accelerate_enable:
                        (
                            actions,
                            actions_logits,
                            values,
                            sequences_actor,
                            sequences_mask_actor,
                            sequences_critic,
                            sequences_mask_critic,
                            action_len_actor,
                            action_len_critic,
                        ) = self.actorcritic.module.generate(
                            states_actor, states_mask_actor, states_critic
                        )
                    else:
                        with torch.autocast(
                            device_type=self.config.trainer.device_type,
                            dtype=torch.float16,
                        ):
                            (
                                actions,
                                actions_logits,
                                values,
                                sequences_actor,
                                sequences_mask_actor,
                                sequences_critic,
                                sequences_mask_critic,
                                action_len_actor,
                                action_len_critic,
                            ) = self.actorcritic.generate(
                                states_actor, states_mask_actor, states_critic
                            )

                    # compute action log probs
                    action_prob = (
                        torch.softmax(actions_logits, dim=-1)
                        .max(dim=-1)
                        .values
                    )
                    actions_log_probs = torch.log(action_prob + self.eps)

                    # get tokenized sequence for the reward models
                    # if the reward model uses the same tokenizer as the actor
                    # then the sequences are already tokenized
                    # otherwise the sequences need to be converted
                    if check_model_family(
                        self.config.actor, self.config.reward
                    ):

                        # they can be directly used since the tokenizer is the
                        # same
                        reward_sequence = sequences_critic
                        reward_mask = sequences_mask_critic
                    else:

                        # convert tokenization
                        if self.accelerate_enable:
                            tokenized_responses = change_tokenization(
                                sequences_actor,
                                self.actorcritic.module.actor.tokenizer,
                                self.reward.tokenizer,
                            )
                        else:
                            tokenized_responses = change_tokenization(
                                sequences_actor,
                                self.actorcritic.actor.tokenizer,
                                self.reward.tokenizer,
                            )

                        # get tokens and mask
                        reward_sequence = tokenized_responses["input_ids"]
                        reward_mask = tokenized_responses["attention_mask"]

                        # move to device
                        if not self.config.reward.load_8bit:
                            reward_sequence = reward_sequence.to(self.device)
                            reward_mask = reward_mask.to(self.device)

                    # compute rewards
                    rewards = self.reward(
                        reward_sequence,
                        reward_mask,
                    )

                    # the interesting rewards are only for the action
                    # TODO: need to use action_len_reward and compute it
                    # before just in case
                    rewards = rewards[:, -action_len_critic:]

                    # the scalar "reward" is the last reward (see reward model)
                    reward = rewards[:, -1]

                    # store memories of the episode and timestep
                    for i in range(states_actor.shape[0]):
                        memories.append(
                            Memory(
                                values[i, :].detach().cpu(),
                                rewards[i, :].detach().cpu(),
                                actions_log_probs[i, :].detach().cpu(),
                                sequences_actor[i, :].detach().cpu(),
                                sequences_mask_actor[i, :].detach().cpu(),
                                sequences_critic[i, :].detach().cpu(),
                                sequences_mask_critic[i, :].detach().cpu(),
                                int(action_len_actor),
                                int(action_len_critic),
                            )
                        )

                    # decode completions to be logged in the conversation log
                    if self.accelerate_enable:
                        completions = [
                            self.actorcritic.module.actor.tokenizer.decode(
                                action
                            )  # noqa 501
                            for action in actions
                        ]
                    else:
                        completions = [
                            self.actorcritic.actor.tokenizer.decode(action)
                            for action in actions
                        ]

                    # remove pad tokens from completions
                    if self.accelerate_enable:
                        completions = [
                            c.replace(
                                self.actorcritic.module.actor.tokenizer.pad_token,  # noqa 501
                                "",
                            )
                            for c in completions
                        ]
                    else:
                        completions = [
                            c.replace(
                                self.actorcritic.actor.tokenizer.pad_token, ""
                            )
                            for c in completions
                        ]

                    # remove eos tokens from completions
                    if self.accelerate_enable:
                        completions = [
                            c.replace(
                                self.actorcritic.module.actor.tokenizer.eos_token,  # noqa 501
                                "",
                            )
                            for c in completions
                        ]
                    else:
                        completions = [
                            c.replace(
                                self.actorcritic.actor.tokenizer.eos_token, ""
                            )
                            for c in completions
                        ]

                    # strange i need to force this?
                    # TODO check how to remove this
                    completions = [c.replace("<pad>", "") for c in completions]

                    # log the memories in the conversation log
                    for i in range(states_actor.shape[0]):
                        self.conversation_log.append(
                            inputs[i],
                            completions[i],
                            reward[i].detach().cpu().item(),
                            cnt_learn_iter,
                        )

                # learn from memories
                if (cnt_timesteps % update_timesteps == 0) and (
                    cnt_timesteps != 0
                ):

                    my_logger.info(f"Len memories {len(memories)}")
                    # self.conversation_log.show(cnt_learn_iter)
                    mean_reward = sum([m.rewards[-1] for m in memories]) / len(
                        memories
                    )
                    my_logger.success(f"Mean Reward: {mean_reward}")
                    self.learn(memories)
                    memories.clear()
                    cnt_timesteps = 0
                    cnt_learn_iter += 1

                    # save conversations for now works only with 1 gpu
                    if (not self.deepspeed_enable) and (
                        not self.accelerate_enable
                    ):
                        self.conversation_log.save()

            # save checkpoints
            if (episode % checkpoint_steps == 0) and (episode != 0):
                self.save_checkpoint(
                    current_epoch=episode,
                    max_epochs=num_episodes,
                )

                # save conversations for now works only with 1 gpu
                if (not self.deepspeed_enable) and (
                    not self.accelerate_enable
                ):
                    self.conversation_log.save()

        # save the models
        if self.accelerate_enable:
            self.actorcritic.module.save()
        else:
            self.actorcritic.save()

        my_logger.success("End RL Training")
