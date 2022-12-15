# **Tensorflow Optimization**

This section contains all the available notebooks that show how to leverage AutoBoost to optimize Tensorflow models.

## Tensorflow API quick view:

``` python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from autoboost import optimize_model

# Load a resnet as example
model = ResNet50()

# Provide an input data for the model    
input_data = [((tf.random.normal([1, 224, 224, 3]),), tf.constant([0]))]

# Run AutoBoost optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="unconstrained"
)

# Try the optimized model
x = tf.random.normal([1, 224, 224, 3])
res_original = model.predict(x)

## Warmup the model
## This step is necessary before the latency computation of the 
## optimized model in order to get reliable results.
# for _ in range(10):
#   optimized_model.predict(x)

res_optimized = optimized_model.predict(x)[0]
```

## Notebooks:
| Notebook                                                                                                                                       | Description                                                            |                                                                                                                                                                                                                                                                                                             |
|:-----------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Keras Resnet50](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/tensorflow/Accelerate_Tensorflow_ResNet50_with_nebullvm.ipynb) | Show how to optimize with Nebullvm a Resnet50 model loaded from keras. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uv3diN_LavkP1Dcc0bkloGrqrn3Yk9ff?usp=sharing) |
