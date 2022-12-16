# **PyTorch Optimization**

This section contains all the available notebooks that show how to leverage AutoBoost to optimize PyTorch models.

## PyTorch API quick view:

``` python
import torch
import torchvision.models as models
from autoboost import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model    
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0]))]

# Run AutoBoost optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = torch.randn(1, 3, 256, 256)

## Warmup the model
## This step is necessary before the latency computation of the 
## optimized model in order to get reliable results.
# for _ in range(10):
#   optimized_model(x)

res = optimized_model(x)
```

## Notebooks:
| Notebook                                                                                                                                                | Description                                                                  |                                                                                                                                                                                                                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Torchvision Resnet50](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/pytorch/Accelerate_PyTorch_ResNet50_with_nebullvm.ipynb) | Show how to optimize with AutoBoost a Resnet50 model loaded from Torchvision. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oNNIRoR94DyV_8KWKmHw2y3wYED3wg3H?usp=sharing) |
| [Accelerate Fast AI Resnet34](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/pytorch/Accelerate_fast_ai_Resnet34_with_nebullvm.ipynb)                                                                                                                         | Show how to optimize with AutoBoost a Resnet34 model loaded from Fast AI.    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xg8RJMV6uvLnJRFRBR1eB2rFmcsF96lS?usp=sharing) |
| [Accelerate Ultralytics YOLOv5](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/pytorch/Accelerate_PyTorch_YOLO_with_nebullvm.ipynb)                                                                                                                       | Show how to optimize with AutoBoost a YOLOv5 model from Ultralytics.          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_vSNm8hGszlLv7rPXChZWLHT-knvz8LS?usp=sharing) |
