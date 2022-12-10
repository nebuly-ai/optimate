# ✒️ Promptify App (WIP)
Effortlessly fine-tune large language and multi-modal models with minimal data and hardware requirements using p-tuning

## 📖 Description
The Promptify App implements p-tuning to use large language and multi-modal models on custom use cases. Fine-tuning a large model (LMs) is typically burdensome in terms of both data and resources required. On the other side, prompt-tuning offers an economical but very effective alternative that provides impressive results when adapting generic LMs to specific use cases.

Promptify combines the efficiency of training a LSTM model to generate task-specific prompts used by a a large model at inference time. The LSTM model is trained on the prompt side, and the generated prompts are inserted at the beginning of the user input to provide task-related information to the large model.

Overall, Promptify allows users to benefit from an effective approach to adapting large language and multi-modal models to your specific use case. Try it out today, and reach out if you have any feedback!
