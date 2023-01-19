# How-to guides

If you’re new to the library, you may want to start with the **Getting started** section.

The user guide here shows more advanced workflows and how to use the library in different ways. We are going to show some examples of more advanced usages of `Speedster`, that we hope will give you a deeper insight of how `Speedster` works. 

In particular, we will overview:

- [Advanced options](#optimize_model-api) of the `optimize_model` API
- [Acceleration suggestions](#acceleration-suggestions)
- [Benchmark API](#benchmark-api)
- [Selecting which device](#selecting-which-device-to-use-cpu-and-gpu) to use for the optimization: CPU and GPU
- [Optimization Time: constrained vs unconstrained](#optimization-time-constrained-vs-unconstrained)
- [Advanced techniques](#advanced-techniques-with-metric-drop) with metric drop
- [Skipping specific compilers/compressors](#skipping-specific-compilerscompressors)
- [Using dynamic shape](#using-dynamic-shape)
- [Custom models](#custom-models)
- [Performances of all the optimization techniques](#performances-of-all-the-optimization-techniques)
- [Set number of threads](#set-number-of-threads)

## Optimize_model API

The `optimize_model` function allows to optimize a model from one of the supported frameworks, and returns an optimized model that can be used with the same interface as the original model.

```python
def optimize_model(
        model: Any,
        input_data: Union[Iterable, Sequence],
        metric_drop_ths: Optional[float] = None,
        metric: Union[str, (...) -> Any, None] = None,
        optimization_time: str = "constrained",
        dynamic_info: Optional[dict] = None,
        config_file: Optional[str] = None,
        ignore_compilers: Optional[List[str]] = None,
        ignore_compressors: Optional[List[str]] = None,
        store_latencies: bool = False,
        device: str = None,
        **kwargs: Any
) -> Any
```

**Arguments**

`model`: Any

The input model, can belong to one of the following frameworks: PyTorch, TensorFlow, ONNX, HuggingFace. In the ONNX case it will be a string (the path to the saved onnx model), in the other cases it will be a torch.nn.Module or a tf.Module.

`input_data`: Iterable or Sequence

Input data to be used for model optimization, which can be one or more data samples. Note that if `optimization_time` is set to "unconstrained," it would be preferable to provide at least 100 data samples to also activate `Speedster` techniques that require data (pruning, etc.). PyTorch DataLoaders and TensorFlow Datasets are fully supported. Otherwise, the data can be entered either as a sequence (data accessible by "element", e.g. `data[i]`) or as an iterable (data accessible with a loop, e.g. `for x in data`). In the case of a input model in PyTorch, TensorFlow and ONNX, a tensor must be passed in the `torch.Tensor`, `tf.Tensor` and `np.ndarray` formats, respectively. Note that each input sample must be a tuple containing a tuple as the first element, the `inputs`, and the `label` as second element. Inputs must be passed as a tuple, even in the case of a single input sample; in such a case, the input tuple will contain only one element. Hugging Face models can take both dictionaries and strings as data samples. In the case of a list of strings passed as input_data, a tokenizer must also be entered as extra arguments with the keyword 'tokenizer'. The strings will then be converted into data samples by Hugging Face tokenizer.

`metric_drop_ths`: float, optional

Maximum drop in your preferred metric (see "metric" section below). No model with a higher error will be accepted, i.e. all optimized model having a larger error with respect to the original one will be discarded, without even considering their possible speed-up. Default: 0.

`metric`: Callable, optional

Metric to be used for estimating the error that may arise from using optimization techniques and for evaluating if the error exceeds the metric_drop_ths and therefore the optimization has to be rejected. metric accepts as input a string, a user-defined metric, or None. Metric accepts a string containing the name of the metric; it currently supports "numeric_precision" and "accuracy". It also supports a user-defined metric that can be passed as a function that takes as input two tuples of tensors, which will be generated from the base model and the optimized model, and their original labels. For more information, see nebullvm.operations.measures.utils.compute_relative_difference and nebullvm.operations.measures.utils.compute_accuracy_drop.  If None is given but a metric_drop_ths is received, the compute_relative_difference metric will be used as the default one. Default: "numeric_precision". 

`optimization_time`: OptimizationTime, optional

The optimization time mode. It can be "constrained" or "unconstrained". In "constrained" mode, Speedster takes advantage only of compilers and precision reduction techniques, such as quantization. "unconstrained" optimization_time allows it to exploit more time-consuming techniques, such as pruning and distillation. Note that most techniques activated in "unconstrained" mode require fine-tuning, and therefore it is recommended that at least 100 samples be provided as input_data. Default: "constrained".

`dynamic_info`: Dict, optional

Dictionary containing dynamic axis information. It should contain as keys both "input" and "output" and as values two lists of dictionaries, where each dictionary represents dynamic axis information for an input/output tensor. The inner dictionary should have an integer as a key, i.e. the dynamic axis (also considering the batch size) and a string as a value giving it a tag, e.g., "batch_size.". Default: None.

`config_file`: str, optional

Configuration file containing the parameters needed to define the CompressionStep in the pipeline. Default: None.

`ignore_compilers`: List[str], optional

List of DL compilers ignored during optimization execution. The compiler name should be one among tvm, tensor RT, openvino, onnxruntime, deepsparse, tflite, bladedisc, torchscript, intel_neural_compressor . Default: None.

`ignore_compressors`: List[str], optional

List of DL compressors ignored during compression execution. The compressor name should be one among sparseml and intel_pruning. Default: None.

`store_latencies`: bool, optional

Parameter thay allows to store the latency for each compiler used by Speedster in a json file, that will be created in the working directory. Default: False.

`device`: str, optional

Device used for inference, it can be cpu or gpu. If not set, gpu will be used if available, otherwise cpu. Default: None.

**Returns: Inference Learner**

Optimized version with the same interface of the input model. For example, optimizing a PyTorch model will return an InferenceLearner object that can be called exactly like a PyTorch model (either with model.forward(input) or model(input)). The optimized model will therefore take as input a torch.Tensors and return a torch.Tensors.

## Acceleration suggestions

If the speedup you obtained with the first optimization with `Speedster` is not enough, we suggest the following actions:

- Include more backends for optimization, i.e. set `--backend all`
- Increase the `metric_drop_ths` by 5%, if possible: see [Optimize_model API](#optimize_model-api)
- Verify that your device is supported by your version of speedster: see [Supported hardware](hardware.md)
- Try to accelerate your model on a different hardware or consider using the CloudSurfer module to automatically understand which is the best hardware for your model: see [CloudSurfer](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/cloud_surfer) module.

## Benchmark API

We can compare the optimized model with the original one in order to measure its speedup. To do this, in `Speedster` we have defined the benchmark function which allows to easily measure the latency of a model:

```python
from nebullvm.tools.benchmark import benchmark

benchmark(model, input_data, device="GPU",n_warmup=50, n_runs=1000)
benchmark(optimized_model, input_data, device="GPU",n_warmup=50, n_runs=1000)
```
This function is currently only available for PyTorch models, but will soon be implemented for the other frameworks.

Alternatively, you can simply implement a loop that makes each model perform n iterations and computes its average latency. In this case, it is important to note that in order to have results as reliable as possible, one must first perform a warm-up of the model, and then calculate the average latency over several iterations, as in the following example in PyTorch:

```python
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device).eval()
example_tensor = torch.randn(1, 3, 224, 224).to(device)
num_iters = 100

# Original model
with torch.no_grad():
    # Warmup
    for _ in range(10):
        model(example_tensor)
    start = time.time()
    for _ in range(num_iters):
        model(example_tensor)
print(f"Original model took {(time.time()-start)/num_iters} seconds/iteration.")

# Optimized model
with torch.no_grad():
    # Warmup
    for _ in range(10):
        optimized_model(example_tensor)
    start = time.time()
    for _ in range(num_iters):
        optimized_model(example_tensor)
print(f"Optimized model took {(time.time()-start)/num_iters} seconds/iteration.")
```

## Selecting which device to use: CPU and GPU

The parameter `device` allows to select which device we want to use for inference. By default, `Speedster` will use the gpu if available on the machine, otherwise it will use cpu. If we are running on a machine with a gpu available and we want to optimize the model for cpu inference, we can use:

```python
from speedster import optimize_model

optimized_model = optimize_model(
  model, input_data=input_data, device="cpu"
)
```

## Optimization Time: constrained vs unconstrained

One of the first options that can be customized in `Speedster` is the `optimization_time` parameter. In order to optimize the model, `Speedster` will try a list of compilers which allow to keep the same accuracy of the original model. In addition to compilers, it can also use other techniques such as pruning, quantization, and other compression techniques which can lead to a little drop in accuracy and may require some time to complete. 

We defined two scenarios:

- **constrained**: only compilers and precision reduction techniques are used, so the compression step (the most time consuming one) is skipped. Moreover, in some cases the same compiler could be available for more than one pipeline, for example tensor RT is available in both PyTorch and ONNX. In the constrained scenario each compiler will be used only once, so if for example we optimize a PyTorch model and tensor RT in the PyTorch pipeline manages to optimize the model, it won't be used again in the ONNX pipeline.

- **unconstrained**: in this scenario `Speedster` will use all the compilers available, even if they appear in more than one pipeline. It also allows the usage of more time consuming techniques such as pruning and distillation. Note that for using many of the sophisticated techniques in the 'unconstrained' optimization, a small fine-tuning of the model will be needed. Thus, we highly recommend to give as input_data at least 100 samples when selecting 'unconstrained' optimization.

## Advanced techniques with metric drop

`Speedster` can use some techniques to compress the model, such as pruning, quantization and distillation. These techniques however can lead to an accuracy drop in the model, because they change its weights and/or the structure of the model itself. 

By default, these techniques are turned off and the optimized model will give the same output as the original one, however they can be activated by setting the `metric_drop_ths` parameter. This parameter represents the maximum reduction accepted in the selected metric, and no model with an higher error will be used, i.e. all optimized models having a larger error compared to the original one will be discarded, without even considering their possible speed-up. 

The default metric used by `Speedster` compares the mean relative difference between the original and the optimized model outputs, otherwise the user can set `metric_drop_ths = "accuracy"` to use the accuracy metric or also provide a custom metric function.

```python
from speedster import optimize_model

# Example 1
# Run Speedster optimization using compression techniques,
# accepting a drop in accuracy up to 2%
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="unconstrained", 
    metric_drop_ths=0.02, 
    metric="accuracy"
)

# Example 2
# Run Speedster optimization using compression techniques, 
# accepting a mean relative difference up to 0.1
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="unconstrained", 
    metric_drop_ths=0.1
)

# Example 3
# Run Speedster optimization using only compilers and quantization, 
# accepting a mean relative difference up to 0.1
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained", 
    metric_drop_ths=0.1
)
```

##  Skipping specific compilers/compressors

The `optimize_model` functions accepts also the parameters `ignore_compilers` and `ignore_compressors`, which allow to skip specific compilers or compressors. For example, if we wanted to skip the `tvm` and `bladedisc` optimizers, we could write:

```python
from speedster import optimize_model

optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    ignore_compilers=["tvm", "bladedisc"]
)

# You can find the list of all compilers and compressors below
# COMPILER_LIST = [
#     "deepsparse",
#     "tensor RT",
#     "torchscript",
#     "onnxruntime",
#     "tflite",
#     "tvm",
#     "openvino",
#     "bladedisc",
#     "intel_neural_compressor",
# ]
# 
# COMPRESSOR_LIST = [
#     "sparseml",
#     "intel_pruning",
# ]
```

## Using dynamic shape

By default, a model optimized with `Speedster` will have a static shape. This means that it can be used in inference only with the same shape of the inputs provided to the `optimize_model` function during the optimization. The dynamic shape however is fully supported, and can be enabled with the `dynamic_info` parameter. 

Let's see an example of a model that takes two inputs, where the batch size must be dynamic, as well as the size on the third and fourth dimensions.

```python
import torch
import torchvision.models as models
from speedster import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [((torch.randn(1, 3, 256, 256),), torch.tensor([0])) for _ in range(100)]

# Set dynamic info
dynamic_info = {
    "inputs": [
        {0: "batch", 2: "dim_image", 3: "dim_image"}
    ],
    "outputs": [
        {0: "batch", 1: "out_dim"}
    ]
}

# Run Speedster optimization in one line of code
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained", 
    dynamic_info=dynamic_info
)
```

## Custom models

`Speedster` is designed to optimize models that take as inputs and return in output only tensors or np.ndarrays (and dictionaries/strings for huggingface). Some models may require instead a custom input, for example a dictionary where the keys are the names of the inputs and the values are the input tensors, or may return a dictionary as output. We can optimize such models with `Speedster` by defining a model wrapper.

Let's take the example of the detectron2 model which takes as input a tuple of tensors but returns a dictionary as output:

```python
 class BaseModelWrapper(torch.nn.Module):
    def __init__(self, core_model, output_dict):
        super().__init__()
        self.core_model = core_model
        self.output_names = [key for key in output_dict.keys()]
    
    def forward(self, *args, **kwargs):
        res = self.core_model(*args, **kwargs)
        return tuple(res[key] for key in self.output_names)


class OptimizedWrapper(torch.nn.Module):
    def __init__(self, optimized_model, output_keys):
        super().__init__()
        self.optimized_model = optimized_model
        self.output_keys = output_keys
    
    def forward(self, *args):
        res = self.optimized_model(*args)
        return {key: value for key, value in zip(self.output_keys, res)}

input_data = [((torch.randn(1, 3, 256, 256)), torch.tensor([0]))]

# Compute the original output of the model (in dict format) 
res = model_backbone(torch.randn(1, 3, 256, 256))

# Pass the model and the output sample to the wrapper
backbone_wrapper = BaseModelWrapper(model_backbone, res)

# Optimize the model wrapper
optimized_model = optimize_model(backbone_wrapper, input_data=input_data)

# Wrap the optimized model with a new wrapper to restore the original model output format
optimized_backbone = OptimizedWrapper(optimized_model, backbone_wrapper.output_names)

```

You can find other examples in the [notebooks](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks) section available on GitHub.

## Performances of all the optimization techniques

`Speedster` internally tries all the techniques available on the target hardware and automatically chooses the fastest one. If you need more details on the inference times of each compiler, you can set the `store_latencies` parameter to `True`. A json file will be created in the working directory, listing all the results of the applied techniques and of the original model itself.

```python
# Run Speedster optimization in one line of code
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    store_latencies=True
)
```

## Set number of threads
When running multiple replicas of the model in parallel, it would be useful for CPU-optimized algorithms to limit the number of threads to use for each model. In `Speedster`, it is possible to set the maximum number of threads a single model can use with the environment variable `NEBULLVM_THREADS_PER_MODEL`. 

For instance, you can run:

```python
export NEBULLVM_THREADS_PER_MODEL = 2
```

for using just two CPU threads per model at inference time and during optimization.