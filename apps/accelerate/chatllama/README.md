# **Open source implementation for LLaMA-based ChatGPT. 15x faster training process than ChatGPT (wip)**

Meta has recently released LLaMA, a collection of foundational large language models ranging from 7 to 65 billion parameters.
LLaMA is creating a lot of excitement because it is smaller than GPT-3 but has better performance. For example, LLaMA's 13B architecture outperforms GPT-3 despite being 10 times smaller. This new collection of fundamental models opens the door to faster inference performance and chatGPT-like real-time assistants, while being cost-effective and running on a single GPU.

However, LLaMA was not fine-tuned for instruction task with a Reinforcement Learning from Human Feedback (RLHF) training process.

The good news is that we introduce `ChatLLaMA`, the first open source implementation of LLaMA based on RLHF:

- A complete open source implementation that enables you to build a ChatGPT-style service based on pre-trained LLaMA models.
- Compared to the original ChatGPT, the training process and single-GPU inference are much faster and cheaper by taking advantage of the smaller size of LLaMA architectures.
- ChatLLaMA has built-in support for DeepSpeed ZERO to speedup the fine-tuning process.
- The library also supports all LLaMA model architectures (7B, 13B, 33B, 65B), so that you can fine-tune the model according to your preferences for training time and inference performance.

If you like the project, please show your support by [leaving a star ⭐](https://github.com/nebuly-ai/nebullvm/stargazers).


<img width="1032" alt="Screen Shot 2023-02-26 at 10 56 13 PM" src="https://user-images.githubusercontent.com/83510798/221439813-5972d029-dae5-4561-ab3d-5a55fa5cde09.png">

Image from [OpenAI’s blog](https://openai.com/blog/chatgpt).


# Get started with ChatLLaMA

> :warning: Please note this code represents the algorithmic implementation for RLHF training process of LLaMA and does not contain the model weights. To access the model weights, you need to apply to Meta's [form](https://forms.gle/jk851eBVbX1m5TAv5).

ChatLLaMA allows you to easily train LLaMA-based architectures in a similar way to ChatGPT, using RLHF.
For example, below is the code to start the training in the case of ChatLLaMA 7B.

```python
from chatllama.rlhf.trainer import RLTrainer
from chatllama.rlhf.config import Config

path = "path_to_config_file.yaml"
config = Config(path=path)
trainer = RLTrainer(config.trainer)
trainer.distillate()
trainer.train()
trainer.training_stats.plot()
```

Note that you should provide Meta's original weights and your custom dataset before starting the fine-tuning process. Alternatively, you can generate your own dataset using LangChain's agents.

```python
python generate_dataset.py
```

# Call for open-source contributions

We have open-sourced the complete code to replicate the ChatLLaMA implementation, opening up the possibility for each user to fine-tune their own personalized ChatLLaMA assistants. The library is in its very early stages. It can be further extended with the following additions:

- Checkpoints with fine-tuned weights
- Optimization techniques for faster inference
- Support for packaging the model into an efficient deployment framework

All developers are invited to join Nebuly's efforts toward more efficient and open ChatGPT-like assistants.

You can participate in the following ways:

1. Submit an issue or PR on GitHub
2. Join our [Discord group](https://discord.gg/77d5kGSa8e) to chat
