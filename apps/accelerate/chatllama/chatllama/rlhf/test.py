import torch

from actor import ActorTrainer
from config import Config
from trainer import RLTrainer
from reward import RewardTrainer


def test_actor_training(path=None, device=None, debug=False):
    config = Config(path=path, device=device, debug=debug)
    trainer = ActorTrainer(config.actor)
    trainer.train()
    trainer.training_stats.plot()


def test_reward_training(path=None, device=None, debug=False):
    device = torch.device("cuda:0")
    config = Config(path=path, device=device, debug=debug)
    trainer = RewardTrainer(config.reward)
    trainer.train()
    trainer.training_stats.plot()


def test_rl_trainig(path=None, device=None, debug=False):
    device = torch.device("cuda:0")
    config = Config(path=path, device=device, debug=debug)
    trainer = RLTrainer(config.trainer)
    trainer.distillate()
    trainer.train()
    trainer.training_stats.plot()


if __name__ == "__main__":
    reward_training = True
    rl_training = False
    actor_training = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # place here the path to the config.yaml file
    config_path = "/home/pierpaolo/Documents/optimapi/ptuning/config.yaml"

    if reward_training:
        test_reward_training(path=config_path, device=device)
    if rl_training:
        test_rl_trainig(path=config_path, device=device)
    if actor_training:
        test_actor_training(path=config_path, device=device)
