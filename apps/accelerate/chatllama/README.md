# **Open source implementation for LLaMA-based ChatGPT training process. Faster and cheaper training than ChatGPT (wip)**

> :warning: Please note this code represents the algorithmic implementation for RLHF training process of LLaMA and does not contain the model weights. To access the model weights, you need to apply to Meta's [form](https://forms.gle/jk851eBVbX1m5TAv5).

Meta has recently released LLaMA, a collection of foundational large language models ranging from 7 to 65 billion parameters.
LLaMA is creating a lot of excitement because it is smaller than GPT-3 but has better performance. For example, LLaMA's 13B architecture outperforms GPT-3 despite being 10 times smaller. This new collection of fundamental models opens the door to faster inference performance and chatGPT-like real-time assistants, while being cost-effective and running on a single GPU.

However, LLaMA was not fine-tuned for instruction task with a Reinforcement Learning from Human Feedback (RLHF) training process.

The good news is that we introduce `ChatLLaMA`, the first open source implementation of RLHF process that leverages LLaMA:

- A complete open source implementation that enables you to build a ChatGPT-style service based on pre-trained LLaMA models.
- Compared to the original ChatGPT, the training process and single-GPU inference are much faster and cheaper by taking advantage of the smaller size of LLaMA architectures.
- ChatLLaMA has built-in support for DeepSpeed ZERO to speedup the fine-tuning process.
- The library also supports all LLaMA model architectures (7B, 13B, 33B, 65B), so that you can fine-tune the model according to your preferences for training time and inference performance.

If you like the project, please show your support by [leaving a star ⭐](https://github.com/nebuly-ai/nebullvm/stargazers).

<img width="1032" alt="Screen Shot 2023-02-26 at 10 56 13 PM" src="https://user-images.githubusercontent.com/83510798/221439813-5972d029-dae5-4561-ab3d-5a55fa5cde09.png">

Image from [OpenAI’s blog](https://openai.com/blog/chatgpt).

# Installation
You can install the package with pip:
```bash
pip install chatllama-py
```
Then you need to install the Llama models cloned from [Meta's repository](https://github.com/facebookresearch/llama):
```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .
```
Follow the instructions in the Llama repository to download the model weights and tokenizer.

## Generate Synthetic Conversations
We provide a script for generating synthetic conversations that can be used to train the ChatLLaMA model. The script is based on LangChain conversation bots which use OpenAI's `davinci-003` model. Note that generating the synthetic conversations requires an OpenAI API key (and a paid subscription).

> :warning: Generating the full dataset with davinci-003 would cost approximately ~200$.

You can add your OpenAI API key as an environment variable by running the following command in your terminal or command prompt (replacing YOUR_API_KEY with your actual API key):

For Linux/Mac:
```
export OPENAI_API_KEY=YOUR_API_KEY
```

For Windows:
```
set OPENAI_API_KEY=YOUR_API_KEY
```

You can generate your own dataset using LangChain's agents.

```python
python generate_dataset.py
```

## Use an Existing Dataset
Alternatively, training can be boostrapped using a pre-existing dataset available on HuggingFace.  High quality candidates are namely the Anthropic HH RLHF and the Stanford Human Preference datasets.

[Anthropic HH RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
This dataset consists of structured question/response pairs with a LLM chatbot that include chosen and rejected responses.

[Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)
This dataset is curated from selected "ask" subreddits and contains questions spanning a wide array of question/answer pairs based on the most upvoted responses.  Unlike HH RLHF, this dataset is not intended to reduce harmfulness by selecting the ideal response by a chatbot but instead weights the most helpful human responses.


# Get started with ChatLLaMA

> :warning: Please note this code represents the algorithmic implementation for RLHF training process of LLaMA and does not contain the model weights. To access the model weights, you need to apply to Meta's [form](https://forms.gle/jk851eBVbX1m5TAv5).

ChatLLaMA allows you to easily train LLaMA-based architectures in a similar way to ChatGPT, using RLHF.
For example, below is the code to start the training in the case of ChatLLaMA 7B.



```python
from chatllama.rlhf.actor import ActorTrainer
from chatllama.rlhf.config import Config
from chatllama.rlhf.reward import RewardTrainer
from chatllama.rlhf.trainer import RLTrainer

# Load config for training
path = "path_to_config_file.yaml"
config = Config(path=path)

# Reward Pre-Training
rw_trainer = RewardTrainer(config.reward)
rw_trainer.distill()
rw_trainer.train()

# Actor Pre-Training
act_trainer = ActorTrainer(config.actor)
act_trainer.train()

# RLHF Training
rlhf_trainer = RLTrainer(config.trainer)
rlhf_trainer.train()
rlhf_trainer.training_stats.plot()
```

Note that you should provide Meta's original weights and your custom dataset before starting the fine-tuning process.
# Call for open-source 


We have open-sourced the complete code to replicate the ChatLLaMA implementation, opening up the possibility for each user to fine-tune their own personalized ChatLLaMA assistants. The library is in its very early stages. It can be further extended with the following additions:

- Checkpoints with fine-tuned weights
- Optimization techniques for faster inference
- Support for packaging the model into an efficient deployment framework

All developers are invited to join Nebuly's efforts toward more efficient and open ChatGPT-like assistants.

You can participate in the following ways:

1. Submit an issue or PR on GitHub
2. Join our [Discord group](https://discord.gg/77d5kGSa8e) to chat

# License
See the [LICENSE](https://github.com/nebuly-ai/nebullvm/blob/main/apps/accelerate/chatllama/LICENSE) file.
