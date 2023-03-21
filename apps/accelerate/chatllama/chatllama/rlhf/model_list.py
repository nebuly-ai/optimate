# llama models
llama_models = ["llama-7B", "llama-13B", "llama-33B", "llama-65B"]

# HF Models
# encoder-decoder models
hf_models_seq_2_seq = [
    "google/flan-t5-xxl",
    "google/flan-t5-xl",
    "google/flan-t5-large",
    "google/flan-t5-base",
]

# decoder only TODO: codegen is still broken
hf_models_causal_lm = [
    "facebook/opt-125m",
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
    "EleutherAI/gpt-j-6B" "gpt2",
    "gpt2-large",
    "gpt2-xl",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-mono",
]

# create a list of all the models from hf
hf_models = hf_models_seq_2_seq + hf_models_causal_lm
