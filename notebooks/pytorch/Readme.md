# **PyTorch Optimization**

This section contains all the available notebooks that show how to leverage nebullvm to optimize PyTorch models.

## PyTorch API quick view:

``` python
import torch
import torchvision.models as models
from nebullvm.api.functions import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model    
input_data = [((torch.randn(1, 3, 256, 256), ), 0)]

# Run nebullvm optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = torch.randn(1, 3, 256, 256)
res = optimized_model(x)
```

## Notebooks:
- Resnet50 from torchvision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aO8_5HiicAcOWbl-JI45RZZureJfMNJ-?usp=sharing)
- Resnet34 from fast ai [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18vh5tVm91hGMTea-924Lbk8YJ-Np45Qb?usp=sharing)
- YOLOv5 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1whd9kIT26EIbvBAJytlU8OsM__uD7RfF?usp=sharing)
- HuggingFace GPT2 and BERT [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z_dbFIfaeED5XcpGcYJkXhE1vxQS4SsO?usp=sharing)
