import json
import os
import random
from collections import deque, namedtuple

import deepspeed
import torch
from accelerate import Accelerator
from beartype import beartype
from beartype.typing import Deque, Tuple, List
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

from chatllama.rlhf.actor import ActorModel
from chatllama.rlhf.config import ConfigReward, ConfigActor, Config
from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.reward import RewardModel, CriticModel
from chatllama.rlhf.utils import TrainingStats, ConversationLog


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


class ActorCritic(torch.nn.Module):
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model
        debug (bool): enable prints for Debugging

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(
        self, actor_config: ConfigActor, critic_config: ConfigReward
    ) -> None:
        super().__init__()
        self.actor = ActorModel(actor_config)

        # check if critic must be initialized from reward model
        ModelLoader.init_critic_from_reward(critic_config)
        self.critic = CriticModel(critic_config)

        self.debug = actor_config.debug

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
            sequences_actor (torch.Tensor): Sequences composed of
                [states, actions] for the actor
            sequence_mask_actor (torch.Tensor): Mask for the sequences
                of the actor
            sequences_critic (torch.Tensor): Sequences composed of
                [states, actions] for the critic
            sequences_mask_critic (torch.Tensor): Mask for the sequences
                of the critic
            action_len_actor (int): Length of the actions in the sequences
                for the actor
            action_len_critic (int): Length of the actions in the sequences
                for the critic

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """
        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        actions_logits = self.actor.forward(
            sequences_actor, sequences_mask_actor
        )

        values = self.critic.forward(sequences_critic, sequences_mask_critic)

        # return only logits and values for the actions taken
        real_actions_logits = actions_logits[:, -action_len_actor:, :]
        real_values = values[:, -action_len_critic:]

        if self.debug:
            print("ActorCritic.forward")
            print("action_len_actor", action_len_actor)
            print("action_len_critic", action_len_critic)
            print("sequences_actor.shape", sequences_actor.shape)
            print("sequences_actor", sequences_actor)
            print("sequences_critic.shape", sequences_critic.shape)
            print("sequences_critic", sequences_critic)
            print("real_action_logits.shape", actions_logits.shape)
            print("real_action_logits", actions_logits)
            print("real_values.shape", values.shape)
            print("real_values", values)

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
            states_actor (torch.Tensor): States for the actor
            states_mask_actor (torch.Tensor): Mask for the states for the
                actor
            states_critic (torch.Tensor): States for the critic

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """

        # generate action sequence from the actor
        actions, sequences_actor = self.actor.generate(
            states_actor, states_mask_actor
        )
        sequences_mask_actor = (
            sequences_actor != self.actor.tokenizer.pad_token_id
        )
        sequences_mask_actor = (
            sequences_mask_actor.to(sequences_actor.device).long().detach()
        )
        action_len_actor = actions.shape[1]

        # generate sequences for the critic from actor sequences
        decoded_actions = self.actor.tokenizer.decode(actions)
        decoded_critic = self.critic.tokenizer.encode(
            decoded_actions, return_tensors="pt"
        )
        sequences_critic = decoded_critic["input_ids"].to(
            sequences_actor.device,
        )
        sequences_mask_critic = (
            decoded_critic["attention_mask"]
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
        if self.debug:
            print("ActorCritic.generate")
            print("actions shape", actions.shape)
            print("actions", actions)
            print("sequence shape", sequences_actor.shape)
            print("sequence", sequences_actor)
            print("actions_logits shape", actions_logits.shape)
            print("actions_logits", actions_logits)
            print("values shape", values.shape)
            print("values", values)

        return (
            actions,
            actions_logits,
            values,
            sequences_actor,
            sequences_mask_actor,
        )


# structure to store the data for each experience
Memory = namedtuple(
    "Memory",
    [
        "states",
        "actions",
        "sequences",
        "values",
        "rewards",
        "actions_log_probs",
        "sequences_mask",
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
        self.device = device

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states.to(self.device),
            self.data[idx].actions.to(self.device),
            self.data[idx].sequences.to(self.device),
            self.data[idx].values.to(self.device),
            self.data[idx].rewards.to(self.device),
            self.data[idx].actions_log_probs.to(self.device),
            self.data[idx].sequences_mask.to(self.device),
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


class RLTrainer:
    """Train the actor-critic model using RL

    Attributes:
        config (Config): Configuration of the trainer
        debug (bool): Debug mode
        actorcritic (ActorCritic): Actor-critic model
        actor_optim (torch.optim): Optimizer for the actor
        critic_optim (torch.optim): Optimizer for the critic
        reward (RewardModel): Reward model
        training_stats (TrainingStats): Class to store training stats
    Methods:
        train: the training loop that calls the learn function after generating
            the experiences.
        learn: Learn from a batch of experiences and update the actor and the
            critic model.
        load_checkpoint: Load the checkpoint of the actor-critic model
        save_checkpoint: Save the checkpoint of the actor-critic model
        generate_user_input: Generate the user input from the inputs
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.debug = config.trainer.debug

        # initialize agent-critic
        self.actorcritic = ActorCritic(config.actor, config.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actorcritic.actor.parameters(), lr=config.trainer.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.actorcritic.critic.parameters(), lr=config.trainer.critic_lr
        )

        # initialize reward model
        self.reward = RewardModel(config.reward)

        # initialize class to store training stats
        self.training_stats = TrainingStats()
        model_folder, _, _ = ModelLoader.get_model_path(
            config,
            is_checkpoint=True,
        )
        path = os.path.join(model_folder, "conversations_log.json")
        self.conversation_log = ConversationLog(path)

        # initialize examples sampler
        self.example_sampler = ExamplesSampler(config.trainer.examples_path)

        # eps
        self.eps = 1e-8

    @beartype
    def save_checkpoint(
        self,
        current_episode: int,
        max_episode: int,
    ) -> None:

        # Save checkpoint for the critic
        print(f"Saving checkpoint for episode {current_episode+1}..")
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config.critic,
            is_checkpoint=True,
            current_epoch=current_episode,
            max_epochs=max_episode,
            max_steps=0,
        )
        if os.path.exists(path):
            os.remove(path)
        torch.save(
            {
                "episode": current_episode,
                "critic_state_dict": self.actorcritic.critic.state_dict(),
                "critic_optim_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

        # Save checkpoint for the actor_rl - use config vs config.actor
        # to distinguish between actor and actor_rl
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=current_episode,
            max_epochs=max_episode,
            max_steps=0,
        )
        if os.path.exists(path):
            os.remove(path)
        torch.save(
            {
                "episode": current_episode,
                "actor_state_dict": self.actorcritic.actor.state_dict(),
                "actor_optim_state_dict": self.actor_optimizer.state_dict(),
                "training_stats": self.training_stats,
            },
            path,
        )

    @beartype
    def load_checkpoint(
        self,
    ) -> int:

        critic_episode = -1
        actor_episode = -1

        # load the critic checkpoint
        print("Looking for checkpoints...")
        path = ModelLoader.check_model_path(
            config=self.config.critic,
            is_checkpoint=True,
            current_epoch=None,
        )
        if path is not None:
            print("Loading ...")
            checkpoint = torch.load(path)
            critic_episode = checkpoint["episode"]
            self.actorcritic.critic.load_state_dict(
                checkpoint["critic_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optim_state_dict"]
            )

        # load the actor checkpoint
        print("Looking for checkpoints...")
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=None,
        )
        if path is not None:
            print("Loading ...")
            checkpoint = torch.load(path)
            actor_episode = checkpoint["episode"]
            self.actorcritic.actor.load_state_dict(
                checkpoint["actor_state_dict"]
            )
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optim_state_dict"]
            )
            self.training_stats = checkpoint["training_stats"]

        if critic_episode == actor_episode:
            # all ok start from next episode
            return critic_episode + 1
        else:
            print(
                f"There are some discrepancies between the checkpoints"
                f"of actor and critic \nactor episode: {actor_episode}"
                f"\n critic episode: {critic_episode}\n"
            )
            return min(critic_episode, actor_episode) + 1

    @beartype
    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print("Start to Learn...")

        # get parameters
        epochs = self.config.trainer.epochs
        actor_eps_clip = self.config.trainer.actor_eps_clip
        critic_eps_clip = self.config.trainer.critic_eps_clip
        beta_s = self.config.trainer.beta_s
        batch_size = self.config.trainer.batch_size
        device = self.config.trainer.device

        # create dataset from memories
        dataloader = DataLoader(
            ExperienceDataset(memories, device), batch_size=batch_size
        )

        # initialize deepspeed for actor
        if self.config.actor.deepspeed_enable:
            if self.config.actor.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if (
                os.path.exists(self.config.actor.deepspeed_config_path)
                is False
            ):
                raise ValueError(
                    f"DeepSpeed config path"
                    f"{self.config.actor.deepspeed_config_path}"
                    f"does not exist"
                )
            (
                actor_model_engine,
                actor_optimizer,
                dataloader,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.actorcritic.actor,
                model_parameters=self.actorcritic.actor.parameters(),
                training_data=dataloader,
                config=self.config.actor.deepspeed_config_path,
            )
            self.actorcritic.actor = actor_model_engine
        # initialize deepspeed for critic
        if self.config.critic.deepspeed_enable:
            if self.config.critic.deepspeed_config_path is None:
                raise ValueError(
                    "DeepSpeed config path is None, but deepspeed is enabled"
                )
            if (
                os.path.exists(self.config.critic.deepspeed_config_path)
                is False
            ):
                raise ValueError(
                    f"DeepSpeed config path"
                    f"{self.config.critic.deepspeed_config_path}"
                    f"does not exist"
                )
            (
                critic_model_engine,
                self.critic_optimizer,
                _,
                _,
            ) = deepspeed.initialize(
                args=None,
                model=self.actorcritic.critic,
                model_parameters=self.actorcritic.critic.parameters(),
                config=self.config.critic.deepspeed_config_path,
            )
            self.actorcritic.critic = critic_model_engine

        # initialize actor accelerate
        if self.config.actor.accelerate_enable is True:
            self.actor_accelerator = Accelerator()
            (
                actor_model,
                self.actor_optimizer,
                dataloader,
            ) = self.actor_accelerator.prepare(
                self.actorcritic.actor, self.actor_optimizer, dataloader
            )
            self.actorcritic.actor = actor_model
        # initialize critic accelerate
        if self.config.critic.accelerate_enable is True:
            self.critic_accelerator = Accelerator()
            (
                critic_model,
                self.critic_optimizer,
            ) = self.critic_accelerator.prepare(
                self.actorcritic.critic,
                self.critic_optimizer,
            )
            self.actorcritic.critic = critic_model

        # train agent-critic
        self.actorcritic.train()
        for epoch in range(epochs):
            for i, (
                states_actor,
                old_actions,
                old_values,
                rewards,
                old_actions_log_probs,
                sequences_actor,
                sequences_mask_actor,
                sequences_critic,
                sequences_mask_critic,
                action_len_actor,
                action_len_critic,
            ) in enumerate(dataloader):

                # print
                print(
                    f"Epoch {epoch+1} of {epochs}",
                    f"Step {i+1} of {int(len(dataloader) / batch_size)}",
                )

                if self.debug:
                    print("RLTrainer.learn()")
                    print("memory states shapes are: ")
                    print("states shape", states_actor.shape)
                    print("old_actions shape", old_actions.shape)
                    print("sequences shape", sequences_actor.shape)
                    print("old_values shape", old_values.shape)
                    print("rewards shape", rewards.shape)
                    print(
                        "old_actions_log_probs shape",
                        old_actions_log_probs.shape,
                    )
                # reshaping rewards to match [b, s] shape
                rewards = rearrange(rewards, "b -> b 1")

                # get actor critic new probabilities and values
                actions_logits, values = self.actorcritic.forward(
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                )

                # get action log prob
                actions_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob * (old_actions_log_probs - actions_log_prob))
                    .sum(dim=-1)
                    .mean()
                )

                # compute PPO Loss -- Whan dimensions are different
                # (especially the values and the probs are
                #  multiplied directly with the reward)
                ratios = (actions_log_prob - old_actions_log_probs).exp()
                advantages = rewards - old_values
                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps
                )
                surr1 = advantages * ratios
                surr2 = (
                    torch.clamp(ratios, 1 - actor_eps_clip, 1 + actor_eps_clip)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2) - beta_s * entropies
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError("Loss is nan")
                print("loss", loss.item())

                if self.debug:
                    print("values", values)
                    print("old_values", old_values)
                    print("rewards", rewards)
                    print("ratios", ratios)
                    print("advantages", advantages)
                    print("entropies", entropies)

                # update actor with loss
                if self.config.actor.deepspeed_enable:
                    actor_model_engine.backward(loss)
                    actor_model_engine.step()
                elif self.config.actor.accelerate_enable:
                    self.actor_optimizer.zero_grad()
                    self.actor_accelerator.backward(loss)
                    self.actor_optimizer.step()
                else:
                    self.actor_optimizer.zero_grad()
                    loss.backward()
                    self.actor_optimizer.step()

                torch.cuda.synchronize(device)

                # compute value loss
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -critic_eps_clip, critic_eps_clip
                )
                value_loss1 = (value_loss_clipped - rewards) ** 2
                value_loss2 = (values - rewards) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError("Value loss is nan")
                print("value_loss", value_loss.item())

                # upate critic with loss
                if self.config.critic.deepspeed_enable:
                    critic_model_engine.backward(value_loss)
                    critic_model_engine.step()
                elif self.config.critic.accelerate_enable:
                    self.critic_optimizer.zero_grad()
                    self.critic_accelerator.backward(loss)
                    self.critic_optimizer.step()
                else:
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()

                self.training_stats.training_loss.append(
                    loss.detach().cpu().item()
                )
                self.training_stats.value_loss.append(
                    value_loss.detach().cpu().item()
                )

        self.actorcritic.eval()
        print("End Learning")

    def train(
        self,
    ) -> None:
        # initialize settings
        num_episodes = self.config.trainer.num_episodes
        max_timesteps = self.config.trainer.max_timesteps
        num_examples = self.config.trainer.num_examples
        update_timesteps = self.config.trainer.update_timesteps
        batch_size = self.config.trainer.batch_size
        checkpoint_steps = self.config.trainer.checkpoint_steps
        device = self.config.trainer.device

        print("Start RL Training")
        # check dimensions consistency
        # at each time step num_examples memories are generated
        number_of_memories_per_learn_iteration = (
            num_examples * update_timesteps
        )
        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % batch_size == 0
        ), "The number of memories must be a multiple of the batch size"
        # the total number of timesteps is
        total_number_of_timesteps = num_episodes * max_timesteps
        # the update_timesteps must be a multiple
        #  of the total number of timesteps
        assert total_number_of_timesteps % update_timesteps == 0, (
            "The number of timesteps (num_episodes*max_timesteps)"
            "must be a multiple of the update_timesteps"
        )

        # initialize memories
        memories = deque([])

        # load checkpoint
        start_episode = self.load_checkpoint()
        # if it is a new training from the start clear the conversation log
        if start_episode == 0:
            self.conversation_log.clear()

        # initialize counters
        cnt_timesteps = 0
        cnt_learn_iter = 0

        # loop over episodes and timesteps
        self.actorcritic.eval()
        for episode in range(start_episode, num_episodes):
            for timestep in range(max_timesteps):

                print(
                    f"Episode: {episode + 1} of {num_episodes}, "
                    f"Timestep: {timestep + 1} of {max_timesteps}",
                )

                # counter used to count timesteps into memory
                cnt_timesteps += 1

                # sample num_examples examples from  example dataset
                inputs = self.example_sampler.sample(num_examples)

                # tokenize examples for the actor
                tok_inputs_act = self.actorcritic.actor.tokenizer(
                    inputs, padding=True, return_tensors="pt"
                )
                if self.debug:
                    print("RLTrainer.train()")
                    print("tokenized inputs actor", tok_inputs_act)

                # states are [batch_size, seq_len_of_states]
                states_actor = tok_inputs_act["input_ids"].to(device)
                states_mask_actor = tok_inputs_act["attention_mask"].to(device)

                # tokenize examples for the critic
                tok_inputs_crt = self.actorcritic.critic.tokenizer(
                    inputs, padding=True, return_tensors="pt"
                )
                if self.debug:
                    print("RLTrainer.train()")
                    print("tokenized inputs critic", tok_inputs_crt)

                # states are [batch_size, seq_len_of_states]
                states_critic = tok_inputs_crt["input_ids"].to(device)

                # generate prompts
                # actions --> output produced by the actor head in response
                #  of the state(input) [batch_size, len_of_actions]
                # actions_logits --> logits of the actions
                # [batch_size, len_of_actions, vocab_size]
                # values --> output produced by the critic for each action
                # [batch_size, len_of_actions]
                # sequence --> (state, actions)
                # [batch_size, len_of_actions + seq_len_of_states] =
                # [batch_size, seq_len]
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

                # from action logits to action log probs
                action_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_probs = torch.log(action_prob + self.eps)

                completions = [
                    self.actorcritic.actor.tokenizer.decode(action)
                    for i, action in enumerate(actions)
                ]
                if self.debug:
                    print("RLTrainer.train()")
                    print("completions:")
                    for i, completion in enumerate(completions):
                        print(i, completion)
                        print("")

                # compute reward for the completion
                # the reward must take into account the answer quality wrt to
                # the initial request given
                # and must be tokenized again
                task_responses = []
                for input, completion in zip(inputs, completions):
                    task_response = input + "\n" + completion
                    task_responses.append(task_response)
                if self.debug:
                    print("RLTrainer.train()")
                    print("task_responses:")
                    for i, task_response in enumerate(task_responses):
                        print(i, task_response)
                        print("")
                tokenized_responses = self.reward.tokenizer(
                    task_responses, padding=True, return_tensors="pt"
                )
                rewards = self.reward.get_reward(
                    tokenized_responses["input_ids"].to(device),
                    tokenized_responses["attention_mask"].to(device),
                )

                # store memories of the episode / timestep
                for i in range(states_actor.shape[0]):
                    memories.append(
                        Memory(
                            *map(
                                lambda x: x.detach().cpu(),
                                (
                                    states_actor[i, :],
                                    actions[i, :],
                                    values[i, :],
                                    rewards[i],
                                    actions_log_probs[i, :],
                                    sequences_actor[i, :],
                                    sequences_mask_actor[i, :],
                                    sequences_critic[i, :],
                                    sequences_mask_critic[i, :],
                                    action_len_actor[i],
                                    action_len_critic[i],
                                ),
                            )
                        )
                    )

                # log the memories in the conversation log
                for i in range(states_actor.shape[0]):
                    self.conversation_log.add_conversation(
                        inputs[i],
                        completions[i],
                        rewards[i].detach().cpu().item(),
                        cnt_learn_iter,
                    )

                # learn from memories
                print(
                    f"Learning counter: {cnt_timesteps} of {update_timesteps}"
                )
                if (cnt_timesteps % update_timesteps == 0) and (
                    cnt_timesteps != 0
                ):
                    # self.conversation_log.show(cnt_learn_iter)
                    self.learn(memories)
                    memories.clear()
                    cnt_timesteps = 0
                    cnt_learn_iter += 1

            # save checkpoints
            if (episode % checkpoint_steps == 0) and (episode != 0):
                self.save_checkpoint(current_episode=episode)
                self.conversation_log.save()

        self.actorcritic.critic.save()
        self.actorcritic.actor.save()
        print("End RL Training")
