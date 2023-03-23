import argparse

from chatllama.rlhf.actor import ActorTrainer
from chatllama.rlhf.config import Config
from chatllama.rlhf.dataset import BaseDataset
from chatllama.rlhf.reward import RewardTrainer
from chatllama.rlhf.trainer import RLTrainer


# Setup argument parser
parser = argparse.ArgumentParser(
    prog="main.py", description="RLHF Training of ChatBots"
)

parser.add_argument("configfile", help="Path to config.yaml file")
parser.add_argument(
    "-t",
    "--type",
    help=(
        "Specify the training type. RL: Training of the model using RL."
        "ACTOR: Training of the actor model. "
        "REWARD: Training of the reward model."
        "RL: The whole pipeline with the three training steps"
    ),
    default="ALL",
    choices=["ALL", "RL", "ACTOR", "REWARD"],
)
parser.add_argument(
    "-a", "--actor", help="Specify actor model by name", default=None
)
parser.add_argument(
    "-r", "--reward", help="Specify reward model by name", default=None
)

# parse arguments
args = parser.parse_args()

# load config.yaml with all the project informations
config = Config(args.configfile)

# overwrite config if specified differently
if args.actor is not None:
    config.actor.model = args.actor
if args.reward is not None:
    config.reward.model = args.reward

# perform the desired training
if args.type == "RL":
    max_seq = min(
        config.actor.max_sequence_length,
        config.reward.max_sequence_length,
        config.critic.max_sequence_length,
    )
    config.actor.max_sequence_length = max_seq
    BaseDataset.clean_dataset(config)
    rlhf_trainer = RLTrainer(config)
    rlhf_trainer.train()
elif args.type == "ACTOR":
    BaseDataset.clean_dataset(config.actor)
    actor_trainer = ActorTrainer(config.actor)
    actor_trainer.train()
elif args.type == "REWARD":
    BaseDataset.clean_dataset(config.reward)
    reward_trainer = RewardTrainer(config.reward)
    reward_trainer.train()
elif args.type == "ALL":
    reward_trainer = RewardTrainer(config.reward)
    reward_trainer.train()
    actor_trainer = ActorTrainer(config.actor)
    actor_trainer.train()
    rlhf_trainer = RLTrainer(config)
    rlhf_trainer.train()
