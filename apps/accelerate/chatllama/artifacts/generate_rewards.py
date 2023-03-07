import argparse
import json

from langchain import OpenAI, LLMChain, PromptTemplate


class ScoreGenerator:
    def __init__(
        self,
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: int,
        reward_template: dict,
    ) -> None:

        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_model = llm_model

        # initialize LLM and LangChain
        openai_llm = OpenAI(
            model_name=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )

        # Customaize your own Reward template by changing the
        # prompt_template
        prompt_template = PromptTemplate(**reward_template)
        self.llm = LLMChain(llm=openai_llm, prompt=prompt_template)

    def distill(
        self,
        dataset_path: str,
    ) -> None:
        """Parse the dataset and assign scores using LLMs
        then save back the dataset with the uploaded scores
        """

        print("Assigning scores to the reward dataset...")

        # load the dataset
        with open(dataset_path, "r") as f:
            train_data = json.load(f)

        # for each element of the dataset, assing a score.
        for i, data in enumerate(train_data):
            if data.get("score", None) is None:

                print("Distilling data", i)
                print("user_input:", data["user_input"])
                print("completion:", data["completion"])
                print("score:", data["score"])

                prompt_tokens = (
                    data["user_input"]
                    + data["completion"]
                    + self.llm.prompt.template
                )
                prompt_len = int(len(prompt_tokens.split(" ")) / 0.75)
                # 80% of the max length as safety margin
                if prompt_len > self.llm_max_tokens * 0.8:
                    print(
                        f"The prompt of the data {i} is too long\n"
                        f"tokens: {prompt_len}\n"
                        f"max_tokens: {self.llm_max_tokens * 0.8}"
                    )
                    continue
                score = self.llm.run(
                    user_input=data["user_input"],
                    completion=data["completion"],
                ).strip()
                # TODO: extract from score the float value with a regex
                score = score.split(" ")[0]
                try:
                    score = float(score)
                except Exception:
                    print(
                        f"The score returned by the LLM for the"
                        f"data, {i}, is not a float float:\n{score}"
                    )
                    continue
                data["score"] = score
                print("score:", data["score"])
        # save the dataset back
        print("Writing the updated dataset back to disk ... ")
        with open(dataset_path, "w") as f:
            json.dump(train_data, f)

        print("Score Assignment Completed")


if __name__ == "__main__":

    REWARD_TEMPLATE = dict(
        template=(
            "Lets pretend that you are a lawyer and you have to"
            "evalaute the following completion task from a given"
            "assigment with a score between 0 and 5 where 0 represents"
            "a bad assignment completion and 5 a perfect completion.\n"
            "You MUST evaluate: text quality, content quality and"
            "coherence.\n"
            "You MUST return only the number that represents your"
            "judgment.\n"
            "The assignement is:\n{user_input}\n"
            "The completion is:\n{completion}\n"
        ),
        input_variables=["user_input", "completion"],
    )

    # Setup argument parser
    parser = argparse.ArgumentParser(
        prog="generate_rewards.py",
        description="Generate rewards using LangChain and LLMs",
    )

    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to be used",
        default="text-davinci-003",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Specify the temperature of the score assignment",
        default=0.1,
    )
    parser.add_argument(
        "-k",
        "--max_tokens",
        help="Specify the max tokens of the score assignement",
        default=50,
    )
    parser.add_argument(
        "-r",
        "--reward_template",
        help="Specify the reward template to be used",
        default=REWARD_TEMPLATE,
    )

    # parse arguments
    args = parser.parse_args()

    score_generator = ScoreGenerator(
        args.model, args.temperature, args.max_tokens, args.reward_template
    )

    score_generator.distill(args.dataset_path)
