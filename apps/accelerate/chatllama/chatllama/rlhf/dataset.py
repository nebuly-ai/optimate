import json
import os
import random

import numpy as np
from beartype.typing import Dict, List, Union
from datasets import load_dataset

from chatllama.rlhf.config import Config, ConfigActor, ConfigReward
from chatllama.rlhf.utils import load_tokenizer, my_logger


ConfigType = Union[Config, ConfigActor, ConfigReward]


class BaseDataset:
    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def sort_conversation(
        conversations: List[Dict],
        only_input: bool = False,
        reverse: bool = True,
        shuffle: bool = True,
    ) -> List[Dict]:
        """Sort the conversations by length of user_input + completion
        or by length of user_input only

        Args:
            conversations (List[Dict]): list of conversations
            only_input (bool, optional): sort by length of user_input only.
                Defaults to False.
            reverse (bool, optional): sort in descending order.
                Defaults to True.
            shuffle (bool, optional): shuffle the dataset leaving only the
                first 100 samples sorted. Defaults to True.

        Returns:
            List[Dict]: sorted list of conversations
        """

        # define the sorting function
        if only_input is True:

            def sort_fun(x):
                return len(x["user_input"])

        else:

            def sort_fun(x):
                return len(x["user_input"]) + len(x["completion"])

        # sort
        conversations = sorted(
            conversations,
            key=sort_fun,
            reverse=reverse,
        )

        # shuffle
        if shuffle is True:
            # keep the first 128 sorted so that for batch <= 128
            # the out-of-memory error occurs only in the first batch iteration
            # (i.e. is the worst case scenario)
            keep_sorted = 128
            conversations = (
                conversations[:keep_sorted]
                + np.random.choice(
                    conversations[keep_sorted:],
                    size=len(conversations[keep_sorted:]),
                    replace=False,
                ).tolist()
            )

        return conversations

    @staticmethod
    def take_n_samples(
        conversations: List[Dict],
        n: int,
    ) -> List[Dict]:
        """Take N samples from the dataset

        Args:
            conversations (List[Dict]): list of conversations
            n (int): number of samples to take randomly

        Returns:
            List[Dict]: list of N samples
        """

        # sample N number of index from 0 to len(conversations)
        indexes = np.random.choice(len(conversations), size=n, replace=False)
        # take the samples
        conversations = [conversations[i] for i in indexes]
        return conversations

    @staticmethod
    def clean_dataset(config: ConfigType):
        """Clean the datasets by removing too long examples
        The Reward Dataset constraints are:
        - user_input + completion < Reward model max sequence length
        The Actor Dataset constraints are:
        - user_input + completion < Actor model max sequence length
        The RLHF Training Dataset constraints are:
        - user_input + min_completion < Actor model max sequence length
        - user_input + min_completion < Critic model max sequence length
        - user_input + min_completion < Reward model max sequence length

        Args:
            config (Config): config object
        """


        if isinstance(config, Config):
            my_logger.info("Start cleaning the dataset for RLHF")
            # constraints
            r_model_max_seq_len = config.reward.max_sequence_length
            a_model_max_seq_len = config.actor.max_sequence_length
            c_model_max_seq_len = config.critic.max_sequence_length
            min_completion = config.actor.min_tokens
            # dataset
            dataset_path = config.trainer.examples_path
            # tokenizers
            r_tokenizer = load_tokenizer(config.reward)
            a_tokenizer = load_tokenizer(config.actor)
            c_tokenizer = load_tokenizer(config.critic)
            # safety tokens
            safety_tokens = config.actor.additonal_prompt_tokens

        elif isinstance(config, ConfigActor):
            my_logger.info("Start cleaning the dataset for Actor")
            # constraint
            a_model_max_seq_len = config.max_sequence_length
            # dataset
            dataset_path = config.train_dataset_path
            # tokenizer
            a_tokenizer = load_tokenizer(config)
            # safety tokens
            safety_tokens = config.additonal_prompt_tokens

        elif isinstance(config, ConfigReward):
            my_logger.info("Start cleaning the dataset for Reward")
            # constraint
            r_model_max_seq_len = config.max_sequence_length
            # dataset
            dataset_path = config.train_dataset_path
            # tokenizer
            r_tokenizer = load_tokenizer(config)

        # if there is the datasets
        if os.path.exists(dataset_path):

            # load the dataset
            with open(dataset_path, "r") as f:
                conversations = json.load(f)

            # sort in desceding order - longest first
            if isinstance(config, Config):
                conversations = BaseDataset.sort_conversation(
                    conversations,
                    only_input=True,
                    reverse=True,
                    shuffle=False,
                )
            else:
                conversations = BaseDataset.sort_conversation(
                    conversations,
                    only_input=False,
                    reverse=True,
                    shuffle=False,
                )
                
            # save orginal length of dataset
            old_len = len(conversations)
                
            # for reward dataset first remove examples that do not have
            # the scores - these check avoids errors in the training 
            # of the reward model when the score is None
            if isinstance(config, ConfigReward):
                cnt = 0
                while cnt < len(conversations):
                    if conversations[cnt]["score"] is None:
                        conversations.pop(cnt)
                    cnt = cnt + 1

            # remove too long examples
            # since datasets are ordered by the length
            # we can remove the first elements until we find
            # an example that is not too long
            while len(conversations) > 0:

                # get the text to be tokenized
                if isinstance(config, Config):
                    text = conversations[0]["user_input"]
                else:
                    text = (
                        conversations[0]["user_input"]
                        + conversations[0]["completion"]
                    )

                # remove elements from RLHF dataset
                if isinstance(config, Config):
                    a_tokens = a_tokenizer.encode(text, truncation=False)
                    r_tokens = r_tokenizer.encode(text, truncation=False)
                    c_tokens = c_tokenizer.encode(text, truncation=False)
                    if (
                        len(a_tokens) + min_completion + safety_tokens + 100
                        > a_model_max_seq_len
                    ):
                        conversations.pop(0)
                    elif (
                        len(r_tokens) + min_completion + safety_tokens + 100
                        > r_model_max_seq_len
                    ):
                        conversations.pop(0)
                    elif (
                        len(c_tokens) + min_completion + safety_tokens + 100
                        > c_model_max_seq_len
                    ):
                        conversations.pop(0)
                    else:
                        break

                # remove elements from Actor dataset
                elif isinstance(config, ConfigActor):
                    tokens = a_tokenizer.encode(text, truncation=False)
                    if len(tokens) + safety_tokens > a_model_max_seq_len:
                        conversations.pop(0)
                    else:
                        break

                # remove elements from Reward dataset
                elif isinstance(config, ConfigReward):
                    tokens = r_tokenizer.encode(text, truncation=False)
                    if len(tokens) > r_model_max_seq_len:
                        conversations.pop(0)
                    else:
                        break

            # if the number of examples has changed print the cleaning
            # stats
            if len(conversations) != old_len:
                my_logger.info(
                    f"Number of examples before cleaning:  {old_len}"
                )
                my_logger.info(
                    f"Number of examples after cleaning: {len(conversations)}"
                )

                # save the new dataset
                with open(dataset_path, "w") as f:
                    json.dump(conversations, f, indent=4)
                my_logger.success("Dataset cleaned")
            else:
                my_logger.success("Dataset is already clean")

        else:
            my_logger.warning(
                f"Dataset not found at {dataset_path}"
                f" Skipping cleaning of the dataset"
            )
            
    @staticmethod
    def augment_reward_dataset(
        reward_conv: List[Dict],
        ):
        """
        Augment the reward dataset with negative examples
        
        Args:
            reward_conv (List[Dict]): list of reward conversations
        """
        
        # shuffle question and answer to generate wrong answers and assing
        # a low score to them - 20% of the dataset
        percentage = 20
        n = int(len(reward_conv) * percentage / 100 / 2)
        data_shuffle = []
        for i in range(n):
            sample = random.choice(reward_conv)
            sample2 = random.choice(reward_conv)
            new_sample1 = {"score": 1,
                        "user_input": sample2["user_input"],
                        "completion": sample["completion"]}
            new_sample2 = {"score": 1,
                        "user_input": sample["user_input"], 
                        "completion": sample2["completion"]}
            data_shuffle.append(new_sample1)
            data_shuffle.append(new_sample2)

        # Create blank answers and assing a low score to them 
        # 10% of the dataset
        percentage = 10
        n = int(len(reward_conv) * percentage / 100 / 2)
        data_blank = []
        for i in range(n):
            sample = random.choice(reward_conv)
            new_sample = {"score": 0.5,
                        "user_input": sample["user_input"],
                        "completion": ""}
            data_blank.append(new_sample)
            new_sample = {"score": 0.5,
                        "user_input": sample["completion"],
                        "completion": "Assistant:",
                        }
            data_blank.append(new_sample)
        
        # swap words in the original completion and assing a low score to them
        # 10% of the dataset
        percentage = 20
        n = int(len(reward_conv) * percentage / 100)
        data_swap = []
        for i in range(n):
            sample = random.choice(reward_conv)
            # swap words order in the completion
            completion = sample["completion"].split(" ")
            # generate list from 0 to len(completion)
            idx = list(range(len(completion)))
            random.shuffle(idx)
            completion = [completion[i] for i in idx]
            completion = " ".join(completion)
            # add spaces back in the completion creating a string from the list
            new_sample = {
                "score": 0.5,
                "user_input": sample["user_input"],
                "completion": completion
                }
            data_swap.append(new_sample)
            
        # add all the augmented data to the reward_conv
        reward_conv.extend(data_shuffle)
        reward_conv.extend(data_blank)
        reward_conv.extend(data_swap)
        
        return reward_conv
            
            
    @staticmethod
    def generate_datasets(
        conversations: List[Dict],
        dataset_folder: str,
        reward_dataset_size: int,
        ):
        """ Generate the datasets for actor reward and rl trainings
        
        Args:
            conversations (List[Dict]): list of conversations
            dataset_folder (str): path to the folder where to save the datasets
            reward_dataset_size (int): size of the reward dataset
        """
        
        # split conversation list in 80 / 20  in two seperate lists
        actor_conv = conversations[:int(len(conversations) * 0.8)]
        rl_conv = conversations[int(len(conversations) * 0.8):]
        
        # augment rl_conv with 10% of its size from actor_conv
        rl_conv.extend(actor_conv[:int(len(rl_conv) * 0.1)])
        
        # create reward_conv sampling randomly from rl_conv reward_dataset_size
        # samples
        reward_conv = random.sample(rl_conv, reward_dataset_size)
        
        # augment reward datasets with wrong completions to create negative 
        # examples
        reward_conv = BaseDataset.augment_reward_dataset(reward_conv)
        
        # sort the lists
        actor_conv = BaseDataset.sort_conversation(
            actor_conv,
            only_input=False,
            reverse=True,
            shuffle=True,
        )
        rl_conv = BaseDataset.sort_conversation(
            rl_conv,
            only_input=True,
            reverse=True,
            shuffle=True,
        )
        reward_conv = BaseDataset.sort_conversation(
            reward_conv,
            only_input=False,
            reverse=True,
            shuffle=True,
        )
        
        # save actor training data
        with open(f"{dataset_folder}/actor_training_data.json", "w") as f:
            json.dump(actor_conv, f, indent=4)
            
         # save actor training data
        with open(f"{dataset_folder}/rlhf_training_data.json", "w") as f:
            json.dump(rl_conv, f, indent=4) 
            
         # save actor training data
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(reward_conv, f, indent=4)  


class StanfordNLPSHPDataset(BaseDataset):
    """Class for Stanford NLP SHP dataset from HuggingFace"""

    def __init__(
        self,
    ) -> None:
        my_logger.info("Download the dataset")
        self.dataset = load_dataset("stanfordnlp/SHP")
        my_logger.success("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """

        # initialize conversations
        conversations = []

        # loop over the dataset
        for i, d in enumerate(data):
            if d["score_A"] > d["score_B"]:
                response = d["human_ref_A"]
            else:
                response = d["human_ref_B"]

            # compose user_input template
            user_input = d["history"].rstrip("\n")
            user_input = "Human: " + d["history"] + "\n\n##\n\n"

            # compose completion template
            completion = "Assistant: " + response
            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }
            conversations.append(conv)

        return conversations

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int,
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        my_logger.info("Generate datasets for RLHF")

        # take the train and test dataset to create the finetuning dataset
        conversations = self.reformat_dataset(self.dataset["train"])
        conversations.extend(self.reformat_dataset(self.dataset["test"]))

        self.generate_datasets(
            conversations,
            dataset_folder,
            number_of_samples,
        )

        my_logger.success("Generation Completed")


class AnthropicRLHFDataset(BaseDataset):
    """Class for Anthropic RLHF dataset from HuggingFace"""

    def __init__(
        self,
    ) -> None:

        my_logger.info("Download the dataset")
        self.dataset = load_dataset("Anthropic/hh-rlhf")
        my_logger.success("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """

        conversations = []
        for _, d in enumerate(data):
            current_conv = d["chosen"]
            split_answer = current_conv.split("Assistant:")

            # take all the list element in split_answer except the last one
            # and joing them with "Assistant:" in a unique string
            previous_convers = split_answer[0]
            for i, s in enumerate(split_answer[1:-1]):
                previous_convers += "Assistant:" + s

            # remove the last characters if they are "\n" from the previous
            # conversation
            previous_convers = previous_convers.rstrip("\n")
            user_input = previous_convers + "\n\n##\n\n"
            user_input = user_input.lstrip("\n")
            completion = "Assistant: " + split_answer[-1]

            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }

            conversations.append(conv)
        return conversations

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int, reverse: bool = True
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        my_logger.info("Generate datasets for RLHF")

        # generate actor and reward dataset
        conversations = self.reformat_dataset(self.dataset["train"])
        conversations.extend(self.reformat_dataset(self.dataset["test"]))

        self.generate_datasets(
            conversations,
            dataset_folder,
            number_of_samples
        )

        my_logger.success("Generation Completed")


class SelfInstructDataset(BaseDataset):
    """Class for SelfInstruct dataset from HuggingFace"""

    def __init__(
        self,
    ) -> None:
        my_logger.info("Download the dataset")
        self.dataset = load_dataset("HuggingFaceH4/self-instruct")
        my_logger.success("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """

        # here do a tiling to reformat the dataset (otherwise is slow)
        def reformat_shard(shard_data):
            rshard = [
                {
                    "user_input": "Human: " + shard_data["prompt"][i],
                    "completion": "Assistant: " + shard_data["completion"][i],
                }
                for i in range(len(shard_data["prompt"]))
            ]
            return rshard

        # number of shards
        n_split = 100

        # shard size
        shard_size = len(data) // n_split

        # initialize the reformatted dataset list
        reformat_data = []

        # loop over the shards
        for i in range(n_split):
            current_shard = data[
                i * shard_size : (i + 1) * shard_size  # noqa E203
            ]
            reformat_data.extend(reformat_shard(current_shard))
        return reformat_data

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int, reverse: bool = True
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        my_logger.info("Generate datasets for RLHF")

        conversations = self.reformat_dataset(self.dataset["train"])
        
        self.generate_datasets(
            conversations,
            dataset_folder,
            number_of_samples
        )
        
        my_logger.success("Generation Completed")
