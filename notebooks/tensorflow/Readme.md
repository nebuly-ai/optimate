# **Tensorflow Optimization**

This section contains all the available notebooks that show how to leverage nebullvm to optimize Tensorflow models.

## Tensorflow API quick view:

```
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from nebullvm.api.functions import optimize_model

# Load a resnet as example
model = ResNet50()

# Provide an input data for the model    
input_data = [((tf.random.normal([1, 224, 224, 3]),), 0)]

# Run nebullvm optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = tf.random.normal([1, 224, 224, 3])
res_original = model.predict(x)
res_optimized = optimized_model.predict(x)[0]
```

## Notebooks:
- Resnet50 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uv3diN_LavkP1Dcc0bkloGrqrn3Yk9ff?usp=sharing)
