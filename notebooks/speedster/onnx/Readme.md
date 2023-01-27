# **ONNX Optimization**

This section contains all the available notebooks that show how to leverage Speedster to optimize ONNX models.

## ONNX API quick view:

```python
import numpy as np
from speedster import optimize_model

# Load a resnet as example
# Model was downloaded from here: 
# https://github.com/onnx/models/tree/main/vision/classification/resnet
model = "resnet50-v1-12.onnx"

# Provide an input data for the model    
input_data = [((np.random.randn(1, 3, 224, 224).astype(np.float32), ), np.array([0]))]

# Run Speedster optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

## Warmup the model
## This step is necessary before the latency computation of the 
## optimized model in order to get reliable results.
# for _ in range(10):
#   optimized_model(x)

res_optimized = optimized_model(x)
```

## Notebooks:
| Notebook                                                                                                                                | Description                                                          |                                                                                                                                                                                                                                                                                                             |
|:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate ONNX Resnet50](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/onnx/Accelerate_ONNX_ResNet50_with_Speedster.ipynb) | Show how to optimize with Speedster a Resnet50 model in ONNX format. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/onnx/Accelerate_ONNX_ResNet50_with_Speedster.ipynb) |
