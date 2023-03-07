import argparse
import json
import os

from datasets import load_dataset


class StanfordNLPSHPDataset:
    def __init__(
        self,
    ) -> None:

        self.dataset = load_dataset("stanfordnlp/SHP")

    def save_dataset(
        self,
        dataset_folder: str,
    ) -> None:

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

        for i, data in enumerate(data["test"]):
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

        with open(f"{dataset_folder}/actor_dataset.json") as f:
            json.dump(conversations, f)
        with open(f"{dataset_folder}/reward_dataset.json") as f:
            json.dump(conversations, f)

        # use the validation part for the rlhf training
        conversations = []
        for i, data in enumerate(self.dataset["validation"]):
            conv = {
                "user_input": data["history"],
            }
            conversations.append(conv)

        with open(f"{dataset_folder}/rlhf_dataset.json") as f:
            json.dump(conversations, f)


class AnthropicRLHF:
    def __init__(
        self,
    ) -> None:

        self.dataset = load_dataset("Anthropic/hh-rlhf")
        print(self.dataset)

    def save_dataset(
        self,
        dataset_folder: str,
    ) -> None:

        # generate actor and reward dataset
        conversations = []
        for i, data in enumerate(self.dataset["train"]):
            current_conv = data["chosen"]

            sections = current_conv.split("Assistant:")
            if len(sections) == 2:
                user_input = sections[0]
                completion = sections[1]
            elif len(sections) == 3:
                user_input = f"{sections[0]}\n" f"Assistant: {sections[1]}"
                completion = sections[2]
            elif len(sections) == 5:
                user_input = (
                    f"{sections[0]}\n"
                    f"Assistant: {sections[1]}"
                    f"{sections[2]}\n"
                )
                completion = sections[2]

            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }
            conversations.append(conv)

        with open(f"{dataset_folder}/actor_dataset.json") as f:
            json.dump(conversations, f)
        with open(f"{dataset_folder}/reward_dataset.json") as f:
            json.dump(conversations, f)

        # rlhf dataset
        conversations = []
        for i, data in enumerate(self.dataset["train"]):
            current_conv = data["chosen"]

            # ABA --> B
            sections = current_conv.split("Assistant:")
            user_input = f"{sections[0]}\n" f"Assistant: {sections[1]}"
            conv = {
                "user_input": user_input,
            }
            conversations.append(conv)

            # A --> B
            sections = current_conv.split("Assistant:")
            conv = {"user_input": sections[0]}
            conversations.append(conv)

        with open(f"{dataset_folder}/rlhf_dataset.json") as f:
            json.dump(conversations, f)


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

    args = parser.parse_args()
    if os.path.exists(args.path):
        os.mkdir(args.path)

    if args.dataset_name == "SHP":
        dataset = StanfordNLPSHPDataset()
        dataset.save_dataset(args.path)

    elif args.dataset_name == "ARLHF":
        dataset = AnthropicRLHF()
        dataset.save_dataset(args.path)
