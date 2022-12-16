# **HuggingFace Optimization**

This section contains all the available notebooks that show how to leverage Speedster to optimize HuggingFace models.

## HuggingFace API quick view:

``` python
from speedster import optimize_model
from transformers import AlbertModel, AlbertTokenizer

# Load Albert as example
model = AlbertModel.from_pretrained("albert-base-v1")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

# Case 1: dictionary input format
text = "This is an example text for the huggingface model."
input_dict = tokenizer(text, return_tensors="pt")

# Run nebullvm optimization
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
#     return_tensors="pt",
#     padding="longest",
#     truncation=True,
# )
# 
# # Run nebullvm optimization
# optimized_model = optimize_model(
#   model, input_data=input_data, tokenizer=tokenizer, tokenizer_args=tokenizer_args
# )
```

## Notebooks:
| Notebook                                                                                                                                                                | Description                                                               |                                                                                                                                                                                                                                                                                                             |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate HuggingFace GPT2](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/pytorch/Accelerate_Hugging_Face_GPT2_with_Speedster.ipynb)            | Show how to optimize with Speedster the GPT2 model from Huggingface.      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TIylOv6JCMXI_MPBdtGKqiTK8RS66ioh?usp=sharing) |
| [Accelerate HuggingFace BERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/pytorch/Accelerate_Hugging_Face_BERT_with_Speedster.ipynb)             | Show how to optimize with Speedster the BERT model from Huggingface.       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dGCVfX7LW-15n5FyCE3wAyX1jtX3jwKJ?usp=sharing) |
| [Accelerate HuggingFace DistilBERT](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/pytorch/Accelerate_Hugging_Face_DistilBERT_with_Speedster.ipynb) | Show how to optimize with Speedster the DistilBERT model from Huggingface. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O6oHzGDKxOoTd2FCLnzqlFND5zFJyXy9?usp=sharing) |

