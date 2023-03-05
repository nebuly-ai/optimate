# **Hugging Face Optimization**

This section contains all the available notebooks that show how to leverage Speedster to optimize Hugging Face models.

Hugging Face hosts models that can use either PyTorch or TensorFlow as backend. Both the backends are supported by Speedster.

## Hugging Face API quick view:

``` python
from speedster import optimize_model
from transformers import AlbertModel, AlbertTokenizer

# Load Albert as example
model = AlbertModel.from_pretrained("albert-base-v1")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

# Case 1: dictionary input format
text = "This is an example text for the huggingface model."
input_dict = tokenizer(text, return_tensors="pt")  # set return_tensors="tf" or "np" for tensorflow models

# Run Speedster optimization
optimized_model = optimize_model(
  model, input_data=[input_dict]
)

## Warmup the model
## This step is necessary before the latency computation of the 
## optimized model in order to get reliable results.
# for _ in range(10):
#   optimized_model(**input_dict)

# Try the optimized model
res = optimized_model(**input_dict)

# # Case 2: strings input format
# input_data = [
#     "This is a test.",
#     "Hi my name is John.",
#     "The cat is on the table.",
# ]
# tokenizer_args = dict(
#     return_tensors="pt",  # set return_tensors="tf" or "np" for tensorflow models
#     padding="longest",
#     truncation=True,
# )
# 
# # Run Speedster optimization
# optimized_model = optimize_model(
#   model, input_data=input_data, tokenizer=tokenizer, tokenizer_args=tokenizer_args
# )
```

## Notebooks:
| Notebook                                                                                                                                                                                     | Description                                                                                     |                                                                                                                                                                                                                                                     |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Hugging Face PyTorch GPT2](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_GPT2_with_Speedster.ipynb)             | Show how to optimize with Speedster the GPT2 model from Hugging Face with PyTorch backend.       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_GPT2_with_Speedster.ipynb)       |
| [Accelerate Hugging Face PyTorch BERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_BERT_with_Speedster.ipynb)             | Show how to optimize with Speedster the BERT model from Hugging Face with PyTorch backend.       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_BERT_with_Speedster.ipynb)       |
| [Accelerate Hugging Face PyTorch DistilBERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_DistilBERT_with_Speedster.ipynb) | Show how to optimize with Speedster the DistilBERT model from Hugging Face with PyTorch backend. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_DistilBERT_with_Speedster.ipynb) |                                                            |
| [Accelerate Hugging Face TensorFlow BERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_TensorFlow_BERT_with_Speedster.ipynb)       | Show how to optimize with Speedster the BERT model from Hugging Face with TensorFlow backend.    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_TensorFlow_BERT_with_Speedster.ipynb)    |
| [Accelerate Hugging Face PyTorch T5](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_T5_with_Speedster.ipynb)                 | Show how to optimize with Speedster the T5 model from Hugging Face with PyTorch backend.         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/huggingface/Accelerate_Hugging_Face_PyTorch_T5_with_Speedster.ipynb)         |
