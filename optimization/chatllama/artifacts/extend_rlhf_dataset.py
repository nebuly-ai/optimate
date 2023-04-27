import os.path

import numpy as np
from langchain import OpenAI, LLMChain, PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _get_template_and_variables(prompt: str, with_examples: bool):
    if with_examples:
        template = prompt + "\n\nExample: {example}"
        variables = ["example"]
    else:
        template = prompt
        variables = []
    return template, variables


def use_langchain_model(
    user_prompt: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    with_examples: bool = False,
) -> LLMChain:
    llm = OpenAI(
        model_name=model_name, temperature=temperature, max_tokens=max_tokens
    )
    template, input_variables = _get_template_and_variables(
        user_prompt, with_examples=with_examples
    )
    prompt_template = PromptTemplate(
        template=template,
        input_variables=input_variables,
    )

    return LLMChain(llm=llm, prompt=prompt_template)


class HuggingFaceChain:
    def __init__(
        self, model_name: str, user_prompt: str, with_examples: bool = False
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt, self.input_variables = _get_template_and_variables(
            user_prompt, with_examples=with_examples
        )

    def run(self, **kwargs):
        prompt = self.prompt.format(**kwargs)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids, max_length=100, num_beams=5, early_stopping=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def use_huggingface_model(
    user_prompt: str,
    model_name: str,
    with_examples: bool = False,
) -> HuggingFaceChain:
    return HuggingFaceChain(
        model_name, user_prompt, with_examples=with_examples
    )


def main():
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model name.",
        default="google/flan-t5-xl",
    )
    parser.add_argument("--templates", type=str, help="Path to templates.")
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument(
        "--data_dir", type=str, help="Path where data are stored"
    )

    args = parser.parse_args()
    model_name = args.model
    templates_path = args.templates
    data_dir = args.data_dir

    with open(os.path.join(data_dir, "rlhf_training_data.json"), "r") as f:
        examples = json.load(f)

    with open(templates_path, "r") as f:
        templates = json.load(f)
    user_prompt = templates.get("rlhf")
    if user_prompt is None:
        raise ValueError("No rlhs template found.")

    if "davinci" in model_name:
        chain = use_langchain_model(
            user_prompt, model_name, with_examples=True
        )
    else:
        if "t5" not in model_name:
            raise ValueError("Only Flan-t5 models are supported for HF.")
        chain = use_huggingface_model(
            user_prompt, model_name, with_examples=True
        )

    for i in range(args.num_prompts):
        example = np.random.choice(examples)
        new_example = chain.run(example=example["user_input"])
        example_dict = {"user_input": new_example}
        examples.append(example_dict)

    with open(os.path.join(data_dir, "rlhf_training_data.json"), "w") as f:
        json.dump(examples, f)


if __name__ == "__main__":
    main()
