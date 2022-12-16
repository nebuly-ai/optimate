# **ONNX Optimization**

This section contains all the available notebooks that show how to leverage AutoBoost to optimize ONNX models.

## ONNX API quick view:

```python
import numpy as np
from autoboost import optimize_model

# Load a resnet as example
# Model was downloaded from here: 
# https://github.com/onnx/models/tree/main/vision/classification/resnet
model = "resnet50-v1-12.onnx"

# Provide an input data for the model    
input_data = [((np.random.randn(1, 3, 224, 224).astype(np.float32), ), np.array([0]))]

# Run AutoBoost optimization
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
| [Accelerate ONNX Resnet50](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/onnx/Accelerate_ONNX_ResNet50_with_nebullvm.ipynb) | Show how to optimize with AutoBoost a Resnet50 model in ONNX format. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18i3q7LASmQfNIT6VSlUKlS22Kd52J_2y?usp=sharing) |