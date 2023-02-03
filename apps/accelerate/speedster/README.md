# 💥 Speedster

`Speedster` is an open-source module designed to speed up AI inference in just a few lines of code. The library automatically applies the best set of SOTA optimization techniques to achieve the maximum inference speed-up (latency, throughput, model size) physically possible on your hardware (single machine).

`Speedster` makes it easy to combine optimization techniques across the whole software to hardware stack, delivering best in class speed-ups. If you like the idea, give us a star to support the project ⭐

![speedster_benchmarks](https://user-images.githubusercontent.com/42771598/212486740-431328f3-f1e5-47bf-b6c9-b6629399ad09.png)

The core `Speedster` workflow consists of 3 steps:


- [x]  **Select**: input your model in your preferred DL framework and express your preferences regarding:
    - Accuracy loss: do you want to trade off a little accuracy for much higher performance?
    - Optimization time: stellar accelerations can be time-consuming. Can you wait, or do you need an instant answer?
- [x]  **Search**: the library automatically tests every combination of optimization techniques across the software-to-hardware stack (sparsity, quantization, compilers, etc.) that is compatible with your needs and local hardware.
- [x]  **Serve**: finally, `Speedster` chooses the best configuration of optimization techniques and returns an accelerated version of your model in the DL framework of your choice (just on steroids 🚀).


# Installation

Install `Speedster` and its base requirements:
```
pip install speedster
```

Then make sure to install all the available deep learning compilers.
```
python -m nebullvm.installers.auto_installer --compilers all
```
> :warning: For **MacOS** with **ARM processors**, please use a conda environment.
> Moreover, if you want to optimize a **PyTorch model**, PyTorch must be pre-installed 
> on your environment before proceeding to the next step, please install it from this 
> [link](https://pytorch.org/get-started/locally/).

For more details on how to install Speedster, please visit our [Installation](https://docs.nebuly.com/Speedster/installation/) guide.

# Quick start

Only one line of code - that’s what you need to accelerate your model! Find below your getting started guide for 4 different input model frameworks:

<details>
<summary>🔥 PyTorch </summary>
    
In this section, we will learn about the 4 main steps needed to optimize PyTorch models:

1) Input your model and data
2) Run the optimization
3) Save your optimized model 
4) Load and run your optimized model in production

```python
import torch
import torchvision.models as models
from speedster import optimize_model, save_model

#1 Provide input model and data (we support PyTorch Dataloaders and custom input, see the docs to learn more)
model = models.resnet50()  
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0])) for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained",
    metric_drop_ths=0.05
)

#3 Save the optimized model
save_model(optimized_model, "model_save_path")
```

Once the optimization is completed, start using the accelerated model (on steroids 🚀) in your DL framework of choice.

```python
#4 Load and run your PyTorch accelerated model in production
from speedster import load_model

optimized_model = load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started with PyTorch optimization](https://docs.nebuly.com/Speedster/getting_started/pytorch_getting_started/).
    
</details>
<details>
<summary>🤗 Huggingface Transformers </summary>
    
In this section, we will learn about the 4 main steps needed to optimize 🤗 Huggingface Transformer models:

1) Input your model and data
2) Run the optimization
3) Save your optimized model 
4) Load and run your optimized model in production

* <details><summary><b>✅ For Decoder-only or Encoder-only architectures (Bert, GPT, etc)</b></summary>

    ```python
    from transformers import AlbertModel, AlbertTokenizer
    from speedster import optimize_model, save_model

    #1a. Provide input model: Load Albert as example
    model = AlbertModel.from_pretrained("albert-base-v1")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

    #1b. Dictionary input format (also string format is accepted, see the docs to learn more)
    text = "This is an example text for the huggingface model."
    input_dict = tokenizer(text, return_tensors="pt")
    input_data = [input_dict for _ in range(100)]

    #2 Run Speedster optimization (if input data is in string format, also the tokenizer 
    # should be given as input argument, see the docs to learn more)
    optimized_model = optimize_model(
        model, 
        input_data=input_data, 
        optimization_time="constrained",
        metric_drop_ths=0.05
    )

    #3 Save the optimized model
    save_model(optimized_model, "model_save_path")
    ```

    Once the optimization is completed, start using the accelerated model (on steroids 🚀) in your DL framework of choice.

    ```python
    #4 Load and run your Huggingface accelerated model in production
    from speedster import load_model

    optimized_model = load_model("model_save_path")

    output = optimized_model(**input_sample)
    ```
    For more details, please visit [Getting Started with HuggingFace optimization](https://docs.nebuly.com/Speedster/getting_started/hf_getting_started/).

    </details>

* <details><summary><b>✅ For Encoder-Decoder architectures (T5 etc)</b></summary>


    ```python
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from speedster import optimize_model, save_model

    #1a. Provide input model: Load T5 as example
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small") 

    #1b. Dictionary input format
    question = "What's the meaning of life?"
    answer = "The answer is:"
    input_dict = tokenizer(question, return_tensors="pt")
    input_dict["decoder_input_ids"] = tokenizer(answer, return_tensors="pt").input_ids
    input_data = [input_dict for _ in range(100)]

    #2 Run Speedster optimization (if input data is in string format, also the tokenizer 
    # should be given as input argument, see the docs to learn more)
    optimized_model = optimize_model(
        model, 
        input_data=input_data, 
        optimization_time="constrained",
        metric_drop_ths=0.05
    )

    #3 Save the optimized model
    save_model(optimized_model, "model_save_path")
    ```

    Once the optimization is completed, start using the accelerated model (on steroids 🚀) in your DL framework of choice.

    ```python
    #4 Load and run your Huggingface accelerated model in production
    from speedster import load_model

    optimized_model = load_model("model_save_path")

    output = optimized_model(**input_sample)
    ```
    For more details, please visit [Getting Started with HuggingFace optimization](https://docs.nebuly.com/Speedster/getting_started/hf_getting_started/).

    </details>
    
</details>
<details>
    
<summary>🌊 TensorFlow/Keras </summary>
    
In this section, we will learn about the 4 main steps needed to optimize TensorFlow/Keras models:

1) Input your model and data
2) Run the optimization
3) Save your optimized model 
4) Load and run your optimized model in production

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from speedster import optimize_model, save_model

#1 Provide input model and data (we support Keras dataset and custom input, see the docs to learn more)
model = ResNet50() 
input_data = [((tf.random.normal([1, 224, 224, 3]),), tf.constant([0])) for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained",
    metric_drop_ths=0.05
)

#3 Save the optimized model
save_model(optimized_model, "model_save_path")
```

Once the optimization is completed, start using the accelerated model (on steroids 🚀) in your DL framework of choice.

```python
#4 Load and run your TensorFlow accelerated model in production
from speedster import load_model

optimized_model = load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started with TensorFlow optimization](https://docs.nebuly.com/Speedster/getting_started/tf_getting_started/).

</details>
<details>
    
<summary> ⚡ ONNX </summary>

In this section, we will learn about the 4 main steps needed to optimize ONNX models:

1) Input your model and data
2) Run the optimization
3) Save your optimized model 
4) Load and run your optimized model in production

```python
import numpy as np
from speedster import optimize_model, save_model

#1 Provide input model and data
# Model was downloaded from here: 
# https://github.com/onnx/models/tree/main/vision/classification/resnet
model = "resnet50-v1-12.onnx" 
input_data = [((np.random.randn(1, 3, 224, 224).astype(np.float32), ), np.array([0])) for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained",
    metric_drop_ths=0.05
)

#3 Save the optimized model
save_model(optimized_model, "model_save_path")
```

Once the optimization is completed, start using the accelerated model (on steroids 🚀) in your DL framework of choice.

```python
#4 Load and run your ONNX accelerated model in production
from speedster import load_model

optimized_model = load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started with ONNX optimization](https://docs.nebuly.com/Speedster/getting_started/onnx_getting_started/).
    
</details>

# **Documentation**

- [Installation](https://docs.nebuly.com/Speedster/installation/)
- [Getting started with PyTorch optimization](https://docs.nebuly.com/Speedster/getting_started/pytorch_getting_started/)
- [Getting started with HuggingFace optimization](https://docs.nebuly.com/Speedster/getting_started/hf_getting_started/)
- [Getting started with TensorFlow optimization](https://docs.nebuly.com/Speedster/getting_started/tf_getting_started/)
- [Getting started with ONNX optimization](https://docs.nebuly.com/Speedster/getting_started/onnx_getting_started/)
- [Key concepts](https://docs.nebuly.com/Speedster/key_concepts/)
- [Notebooks](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks/speedster)
- [Advanced options](https://docs.nebuly.com/Speedster/advanced_options/)
- [Benchmarks](https://docs.nebuly.com/Speedster/benchmarks/)


# **Key concepts**

`Speedster`'s design reflects our mission to automatically master each and every existing AI acceleration techniques to deliver the **fastest AI ever**. As a result, `Speedster` leverages available enterprise-grade open-source optimization tools. If these tools and  communities already exist, and are distributed under a permissive license (Apache, MIT, etc), we integrate them and happily contribute to their communities. However, many tools do not exist yet, in which case we implement them and open-source the code so that our community can benefit from it.

`Speedster` is shaped around **4 building blocks** and leverages a modular design to foster scalability and integration of new acceleration components across the software to hardware stack.

- [x]  **Converter:** converts the input model from its original framework to the framework backends supported by `Speedster`, namely PyTorch, ONNX and TensorFlow. This allows the Compressor and Compiler modules to apply any optimization technique to the model.
- [x]  **Compressor:** applies various compression techniques to the model, such as pruning, knowledge distillation, or quantization-aware training.
- [x]  **Compiler:** converts the compressed models to the intermediate representation (IR) of the supported deep learning compilers. The compilers apply both post-training quantization techniques and graph optimizations, to produce compiled binary files.
- [x]  **Inference Learner:** takes the best performing compiled model and converts it back into the same interface as the original input model.

![speedster_blocks](https://user-images.githubusercontent.com/42771598/213177175-a76908a2-5eef-4e82-9d54-0fc812131463.png)

The **compressor** stage leverages the following open-source projects:

- [Intel/neural-compressor](https://github.com/intel/neural-compressor): targeting to provide unified APIs for network compression technologies, such as low precision quantization, sparsity, pruning, knowledge distillation, across different deep learning frameworks to pursue optimal inference performance.
- [SparseML](https://github.com/neuralmagic/sparseml): libraries for applying sparsification recipes to neural networks with a few lines of code, enabling faster and smaller models.

The **compiler stage** leverages the following open-source projects:

- [Apache TVM](https://github.com/apache/tvm): open deep learning compiler stack for cpu, gpu and specialized accelerators.
- [BladeDISC](https://github.com/alibaba/BladeDISC): end-to-end Dynamic Shape Compiler project for machine learning workloads.
- [DeepSparse](https://github.com/neuralmagic/deepsparse): neural network inference engine that delivers GPU-class performance for sparsified models on CPUs.
- [OpenVINO](https://github.com/openvinotoolkit/openvino): open-source toolkit for optimizing and deploying AI inference.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): cross-platform, high performance ML inferencing and training accelerator
- [TensorRT](https://github.com/NVIDIA/TensorRT): C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.
- [TFlite](https://github.com/tensorflow/tflite-micro) and [XLA](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla): open-source libraries to accelerate TensorFlow models.



# **Community**
We’re developing `Speedster` for and together with our community, so plase get in touch on GitHub or Discord. 

• **[GitHub issues](https://github.com/nebuly-ai/nebullvm/issues)**: suggest new acceleration components, request new features, and report bugs and improvements.

• **[Discord](https://discord.gg/RbeQMu886J)**: learn about AI acceleration, share exciting projects and hang out with our global community.

The best way to get started is to pick a good-first issue. Please read our [contribution guidelines](https://docs.nebuly.com/contributions/) for a deep dive on how to best contribute to our project!

Don't forget to leave a star ⭐ to support the project and happy acceleration 🚀
