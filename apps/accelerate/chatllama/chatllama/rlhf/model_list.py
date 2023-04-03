# llama models
llama_models = ["llama-7B", "llama-13B", "llama-33B", "llama-65B"]

# HF Models
# encoder-decoder models TODO: still not supported
hf_models_seq_2_seq = [
    "google/flan-t5-xxl",
    "google/flan-t5-xl",
    "google/flan-t5-large",
    "google/flan-t5-base",
    "google/flan-t5-small",
]

# decoder only TODO: codegen is still broken
hf_models_causal_lm = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-11b",
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "bigscience/bloom-560m",
    "bigscience/bloomz-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloomz-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloomz-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloomz-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloomz-7b1",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/gpt-j-6B",
    "gpt2",
    "gpt2-large",
    "gpt2-xl",
    "benjamin/gerpt2",
    "benjamin/gerpt2-large",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-mono",
    "cerebras/Cerebras-GPT-111M",
    "cerebras/Cerebras-GPT-256M",
    "cerebras/Cerebras-GPT-590M",
    "cerebras/Cerebras-GPT-1.3B",
    "cerebras/Cerebras-GPT-2.7B",
    "cerebras/Cerebras-GPT-6.7B",
    "cerebras/Cerebras-GPT-13B",
]

# create a list of all the models from hf
hf_models = hf_models_seq_2_seq + hf_models_causal_lm
