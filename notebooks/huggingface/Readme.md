# **HuggingFace Optimization**

This section contains all the available notebooks that show how to leverage nebullvm to optimize HuggingFace models.

## HuggingFace API quick view:

``` python
from nebullvm.api.functions import optimize_model
from transformers import AlbertModel, AlbertTokenizer

# Load Albert as example
model = AlbertModel.from_pretrained("albert-base-v1")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

# Case 1: dictionary input format
text = "This is an example text for the huggingface model."
input_dict = tokenizer(text, return_tensors="pt")

# Run nebullvm optimization
optimized_model = optimize_model(
  model, input_data=input_data
)

# Try the optimized model
res = optimized_model(**input_dict)

# # Case 2: strings input format
# input_data = [
#     "This is a test.",
#     "Hi my name is John.",
#     "The cat is on the table.",
# ]
# tokenizer_args = dict(
#     return_tensors="pt"
# )
# 
# # Run nebullvm optimization
# optimized_model = optimize_model(
#   model, input_data=input_data, tokenizer=tokenizer, tokenizer_args=tokenizer_args
# )
```

## Notebooks:
| Notebook                                                                                                                                                | Description                                                                                                 |                                                                                                                                                                                                                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate HuggingFace GPT2 and BERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/pytorch/Accelerate_Hugging_Face_GPT2_and_BERT_with_nebullvm.ipynb)                                                                                                                | Show how to optimize with Nebullvm transformers models such as BERT and GPT2 model loaded from Huggingface. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z_dbFIfaeED5XcpGcYJkXhE1vxQS4SsO?usp=sharing) |
