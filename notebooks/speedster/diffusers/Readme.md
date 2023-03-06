# **Diffusers Optimization**

This section contains all the available notebooks that show how to leverage Speedster to optimize Diffusers models.

## Diffusers API quick view:

``` python
import torch
from speedster import optimize_model
from diffusers import StableDiffusionPipeline


# Load Stable Diffusion 1.4 as example
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # On GPU we load by default the model in half precision, because it's faster and lighter.
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Create some example input data
input_data = [
    "a photo of an astronaut riding a horse on mars",
    "a monkey eating a banana in a forest",
    "white car on a road surrounded by palm trees",
    "a fridge full of bottles of beer",
    "madara uchiha throwing asteroids against people"
]

# Run Speedster optimization
optimized_model = optimize_model(
    model=pipe,
    input_data=input_data,
    optimization_time="unconstrained",
    ignore_compilers=["torch_tensor_rt"],  # TensorRT from the torch pipeline has some issues, so we are going to skip it
    metric_drop_ths=0.1,
)

# Try the optimized model
res = optimized_model(test_prompt).images[0]
```

## Notebooks:
| Notebook                                                                                                                                                                             | Description                                                                        |                                                                                                                                                                                                                                             |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Diffusers Stable Diffusion 1.4](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/diffusers/Accelerate_StableDiffusion_1_4_with_Speedster.ipynb) | Show how to optimize with Speedster the Stable Diffusion 1.4 model from Diffusers. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/diffusers/Accelerate_StableDiffusion_1_4_with_Speedster.ipynb) |