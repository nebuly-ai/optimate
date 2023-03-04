import llama_model
from rlhf.config import Config
from rlhf.reward import RewardTrainer
from rlhf.actor import ActorTrainer
from rlhf.trainer import RLTrainer

def main():
    # setup model parallel envirorment 
    # llama_model.setup_model_parallel()
    
    # setup config
    config_path = "/home/pierpaolo/nebullvm/apps/accelerate/chatllama/chatllama/config/config.yaml"
    config = Config(config_path)
    reward_trainer = RewardTrainer(config.reward)
    actor_trainer = ActorTrainer(config.actor)
    rlhf_trainer = RLTrainer(config)
    
    print("######### Distillation #########")
    if config.reward.llm_enable:
        reward_trainer.distill()
        # reload the trainer with the distilled dataset
        reward_trainer = RewardTrainer(config.reward)
    print("######### End Distillation #########")
    
    print("######### Reward Trainer #########")
    reward_trainer.train()
    print("######### Reward Training Completed #########")
    
    print("######### Actor Trainer #########")
    actor_trainer.train()
    print("######### Actor Training Completed #########")
    
    print("########## RLHF ##########")
    rlhf_trainer.train()
    print("########## RLHF Completed ##########")
    

if __name__ == "__main__":
    main()