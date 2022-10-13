# **ONNX Optimization**

This section contains all the available notebooks that show how to leverage nebullvm to optimize ONNX models.

## ONNX API quick view:

```python
import numpy as np
from nebullvm.api.functions import optimize_model

# Load a resnet as example
# Model was downloaded from here: 
# https://github.com/onnx/models/tree/main/vision/classification/resnet
model = "resnet50-v1-12.onnx"

# Provide an input data for the model    
input_data = [((np.random.randn(1, 3, 224, 224).astype(np.float32), ), 0)]

# Run nebullvm optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
res_optimized = optimized_model(x)
```

## Notebooks:
| Notebook                                                                               | Description                                                                                                |                                                                                                                                                                                                                                                                                                             |
|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Resnet50](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/onnx/Accelerate_ONNX_ResNet50_with_nebullvm.ipynb) | Show how to optimize with Nebullvm a Resnet50 model.                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-62bwFaxXAHBi5GOSijyle9vB0WNtVXs?usp=sharing) |
