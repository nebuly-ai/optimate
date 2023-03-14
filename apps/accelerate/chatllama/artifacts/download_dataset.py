import argparse
import json
import os
import re

import numpy as np

from datasets import load_dataset


class StanfordNLPSHPDataset:
    def __init__(
        self,
    ) -> None:
        print("Download the dataset")
        self.dataset = load_dataset("stanfordnlp/SHP")
        print("Download Completed")

    def save_dataset(
        self,
        dataset_folder: str,
        number_of_samples: int,
    ) -> None:

        print("Generate datasets for RLHF")

        # TODO: score in the dataset are not used until now
        # use the train and test dataset to create the finetuning dataset
        # for the actor and the reward model
        # (the second one is rewarded by davinci)
        conversations = []
        for i, data in enumerate(self.dataset["train"]):
            if data["score_A"] > data["score_B"]:
                response = data["human_ref_A"]
            else:
                response = data["human_ref_B"]
            conv = {
                "user_input": data["history"],
                "completion": response,
                "score": None,
            }
            conversations.append(conv)

        for i, data in enumerate(self.dataset["test"]):
            if data["score_A"] > data["score_B"]:
                response = data["human_ref_A"]
            else:
                response = data["human_ref_B"]
            conv = {
                "user_input": data["history"],
                "completion": response,
                "score": None,
            }
            conversations.append(conv)

        # sort conversations by length of user_input + completion
        conversations = sorted(
            conversations,
            key=lambda x: len(x["user_input"]) + len(x["completion"]),
            reverse=True,
        )

        with open(f"{dataset_folder}/actor_training_data.json", "w") as f:
            json.dump(conversations, f)

        # sample N number of index from 0 to len(conversations)
        indexes = np.random.choice(
            len(conversations), size=number_of_samples, replace=False
        )
        # sort conversations by length of user_input + completion
        conversations = [conversations[i] for i in indexes]
        conversations = sorted(
            conversations,
            key=lambda x: len(x["user_input"]) + len(x["completion"]),
            reverse=True,
        )
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(conversations, f)

        # use the validation part for the rlhf training
        conversations = []
        for i, data in enumerate(self.dataset["validation"]):
            conv = {
                "user_input": data["history"],
            }
            conversations.append(conv)
        # sort conversations by length of user_input + completion
        conversations = sorted(
            conversations, key=lambda x: len(x["user_input"]), reverse=True
        )

        with open(f"{dataset_folder}/rlhf_training_data.json", "w") as f:
            json.dump(conversations, f)

        print("Generation Completed")


class AnthropicRLHF:
    def __init__(
        self,
    ) -> None:

        print("Download the dataset")
        self.dataset = load_dataset("Anthropic/hh-rlhf")
        print("Download Completed")

    def save_dataset(
        self,
        dataset_folder: str,
        number_of_samples: int,
    ) -> None:

        print("Generate datasets for RLHF")

        # generate actor and reward dataset
        conversations = []
        for i, data in enumerate(self.dataset["train"]):
            current_conv = data["chosen"]

            sections = re.split("Assistant:|User:", current_conv)
            if len(sections) == 2:
                user_input = sections[0]
                completion = sections[1]
            elif len(sections) == 4:
                user_input = (
                    f"Human:{sections[0]}\n"
                    f"Assistant: {sections[1]}"
                    f"Human:{sections[2]}\n"
                )
                completion = sections[3]
            elif len(sections) == 6:
                user_input = (
                    f"Human:{sections[0]}\n"
                    f"Assistant: {sections[1]}"
                    f"Human:{sections[2]}\n"
                    f"Assistant: {sections[3]}\n"
                    f"Human:{sections[4]}\n"
                )
                completion = sections[5]
            else:
                continue

            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }
            conversations.append(conv)

        # sort conversations by length of user_input + completion
        conversations = sorted(
            conversations,
            key=lambda x: len(x["user_input"]) + len(x["completion"]),
            reverse=True,
        )

        with open(f"{dataset_folder}/actor_training_data.json", "w") as f:
            json.dump(conversations, f)

        # sample N number of index from 0 to len(conversations)
        indexes = np.random.choice(
            len(conversations), size=number_of_samples, replace=False
        )
        conversations = [conversations[i] for i in indexes]
        # sort conversations by length of user_input + completion
        conversations = sorted(
            conversations,
            key=lambda x: len(x["user_input"]) + len(x["completion"]),
            reverse=True,
        )
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(conversations, f)

        # rlhf dataset
        conversations = []
        for i, data in enumerate(self.dataset["train"]):
            current_conv = data["chosen"]

            sections = re.split("Assistant:|User:", current_conv)
            if len(sections) >= 2:
                user_input = sections[0]
                completion = sections[1]
                conv = {
                    "user_input": user_input,
                    "completion": completion,
                }
                conversations.append(conv)
            if len(sections) >= 4:
                user_input = (
                    f"Human:{sections[0]}\n"
                    f"Assistant: {sections[1]}"
                    f"Human:{sections[2]}\n"
                )
                completion = sections[3]
                conv = {
                    "user_input": user_input,
                    "completion": completion,
                }
                conversations.append(conv)
            if len(sections) == 6:
                user_input = (
                    f"Human:{sections[0]}\n"
                    f"Assistant: {sections[1]}"
                    f"Human:{sections[2]}\n"
                    f"Assistant: {sections[3]}\n"
                    f"Human:{sections[4]}\n"
                )
                completion = sections[5]
                conv = {
                    "user_input": user_input,
                    "completion": completion,
                }
                conversations.append(conv)

        # sort conversations by length of user_input + completion
        conversations = sorted(
            conversations, key=lambda x: len(x["user_input"]), reverse=True
        )

        with open(f"{dataset_folder}/rlhf_training_data.json", "w") as f:
            json.dump(conversations, f)

        print("Generation Completed")


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(
        prog="generate_rewards.py",
        description="Generate rewards using LangChain and LLMs",
    )

    parser.add_argument(
        "dataset_name",
        help="dataset name it can be. SSHP: stanfordnlp/SHP or ",
        choices=["SHP", "ARLHF"],
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Specify the path for the dataset",
        default="./datasets",
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        help="Specify the number of samples for the reward dataset",
        default=200,
    )

    args = parser.parse_args()
    if os.path.exists(args.path) is False:
        os.mkdir(args.path)

    try:
        n_samples = int(args.number_of_samples)
    except ValueError:
        raise ValueError("Number of samples should be an integer")

    if args.dataset_name == "SHP":
        dataset = StanfordNLPSHPDataset()
        dataset.save_dataset(args.path, n_samples)

    elif args.dataset_name == "ARLHF":
        dataset = AnthropicRLHF()
        dataset.save_dataset(
            args.path,
            n_samples,
        )
