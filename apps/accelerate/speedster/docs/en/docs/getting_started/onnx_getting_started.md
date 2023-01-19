# Getting started with ONNX optimization
In this section, we will learn about the 4 main steps needed to optimize your ONNX models:

1. [Input your model and data]()
2. [Run the optimization]()
3. [Save your optimized model]()
4. [Load and run your optimized model in production]()

## 1) Input model and data

!!! info
    In order to optimize a model with `Speedster`, first you should input the model you want to optimize and load some sample data that will be needed to test the optimization performances (latency, throughput, accuracy loss, etc). 

```python
import numpy as np

# Load a resnet as example
# Model was downloaded from here: 
# https://github.com/onnx/models/tree/main/vision/classification/resnet
model = "resnet50-v1-12.onnx"

# Provide input data for the model    
input_data = [((np.random.randn(1, 3, 224, 224).astype(np.float32), ), np.array([0])) for _ in range(100)]
```

Now your input model and data are ready, you can move on to [Run the optimization]() section 🚀.

## 2) Run the optimization
Once the `model` and `input_data` have been defined, everything is ready to use Speedster's `optimize_model` function to optimize your model. 

The function accepts as input the model in your preferred framework (ONNX in this case) and returns the accelerated version of your model:

``` python
from speedster import optimize_model

# Run Speedster optimization
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained"
)
```

Internally, `Speedster` tries to use all the compilers and optimization techniques at its disposal along the software to hardware stack to optimise the model. From these, it will choose the ones with the lowest latency on the specific hardware. 

At the end of the optimization you are going to see the results of the optimization in a summary table like the following:

![speedster_benchmarks](https://files.gitbook.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FOfr7q1XcUfo7iYMV6A0C%2Fuploads%2FatlsMFyCdXniR6c7kyJG%2FMicrosoftTeams-image%20(17).png?alt=media&token=2bcf4e93-91b1-4345-bd70-27fc991ea2a1)

If the speedup you obtained is good enough for your application, you can move to the [Save your optimized model]() section to save your model and use it in production.

If you want to squeeze out even more acceleration out of the model, please see the [Optimize_model API]() section. Consider if in your application you can trade off a little accuracy for much higher performance and use the `metric`, `metric_drop_ths` and `optimization_time` arguments accordingly.

## 3) Save your optimized model
After accelerating the model, it can be saved using the save method:

```python
optimized_model.save("model_save_path")
```

Now you are all set to use your optimized model in production. To explore how to do it, see the [Load and run your optimized model]() in production section.

## 4) Load and run your optimized model in production
Once the optimized model has been saved,  it can be loaded in the following way:
```python
from nebullvm.operations.inference_learners.base import LearnerMetadata

optimized_model = LearnerMetadata.read("model_save_path").load_model("model_save_path")
```

The optimized model can be used for accelerated inference in the same way as the original model:

```python
# Use the accelerated version of your ONNX model in production
output = optimized_model(input_sample)
```

!!! info
    The first 1-2 inferences could be a bit slower than expected because some compilers still perform some optimizations during the first iterations. After this warm-up time, the next ones will be faster than ever.
