# 💥 Speedster

`Speedster` is an open-source module designed to speed up AI inference in just a few lines of code. The library automatically apply the best set of SOTA optimization techniques to achieve the maximum inference speed-up that is physically possible on your hardware.

We are building a new AI inference acceleration product leveraging state-of-the-art open-source optimization tools enabling the optimization of the whole software to hardware stack. If you like the idea, give us a star to support the project ⭐

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

Then make sure to install the deep learning compilers to leverage during the optimization:
```
python -m nebullvm.installers.auto_installer --backends all --compilers all
```
> :warning: For **MacOS** with **ARM processors**, please use a conda environment.
> Moreover, if you want to optimize a **PyTorch model**, PyTorch must be pre-installed 
> on your environment before proceeding to the next step, please install it from this 
> [link](https://pytorch.org/get-started/locally/).

For more details on the installation step, please visit [Installation](https://docs.nebuly.com/modules/speedster/installation).


# Quick start

Only a single line of code is needed to get your accelerated model! Find below your getting started guide depending on your input model framework:

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
from speedster import optimize_model

#1 Provide input model and data
model = models.resnet50()  
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0])) for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="constrained"
)

#3 Save the optimized model
optimized_model.save("model_save_path")
```

After the optimization, you can start using your accelerated model in the DL framework of your choice (just on steroids 🚀).

```python
#4 Load and run your PyTorch accelerated model in production
from nebullvm.operations.inference_learners.base import LearnerMetadata

optimized_model = LearnerMetadata.read("model_save_path").load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started](https://docs.nebuly.com/modules/speedster/getting-started) and [How-to guides](https://docs.nebuly.com/modules/speedster/how-to-guides).
    
</details>
<details>
<summary>🤗 Huggingface Transformers </summary>
    
In this section, we will learn about the 4 main steps needed to optimize 🤗 Huggingface Transformer models:

1) Input your model and data
2) Run the optimization
3) Save your optimized model 
4) Load and run your optimized model in production

```python
from transformers import AlbertModel, AlbertTokenizer

#1a. Provide input model: Load Albert as example
model = AlbertModel.from_pretrained("albert-base-v1")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")

#1b. Dictionary input format (also string format is accepted, see the docs to learn more)
text = "This is an example text for the huggingface model."
input_dict = tokenizer(text, return_tensors="pt")
input_data = [input_dict for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="constrained"
)

#3 Save the optimized model
optimized_model.save("model_save_path")
```

After the optimization, you can start using your accelerated model in the DL framework of your choice (just on steroids 🚀).

```python
#4 Load and run your PyTorch accelerated model in production
from nebullvm.operations.inference_learners.base import LearnerMetadata

optimized_model = LearnerMetadata.read("model_save_path").load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started](https://docs.nebuly.com/modules/speedster/getting-started) and [How-to guides](https://docs.nebuly.com/modules/speedster/how-to-guides).
    
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

#1 Provide input model and data
model = ResNet50() 
input_data = [((tf.random.normal([1, 224, 224, 3]),), tf.constant([0])) for _ in range(100)]

#2 Run Speedster optimization
optimized_model = optimize_model(
  model, input_data=input_data, optimization_time="constrained"
)

#3 Save the optimized model
optimized_model.save("model_save_path")
```

After the optimization, you can start using your accelerated model in the DL framework of your choice (just on steroids 🚀).

```python
#4 Load and run your PyTorch accelerated model in production
from nebullvm.operations.inference_learners.base import LearnerMetadata

optimized_model = LearnerMetadata.read("model_save_path").load_model("model_save_path")

output = optimized_model(input_sample)
```
For more details, please visit [Getting Started](https://docs.nebuly.com/modules/speedster/getting-started) and [How-to guides](https://docs.nebuly.com/modules/speedster/how-to-guides).

</details>
<details>
    <summary> ⚡ ONNX </summary>
    Foldable Content[enter image description here][1]
</details>

# **Documentation**

- [Installation](https://docs.nebuly.com/modules/speedster/installation)
- [Getting started](https://docs.nebuly.com/modules/speedster/getting-started)
- [Key concepts](https://docs.nebuly.com/modules/speedster/key-concepts)
- [Notebooks](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks)
- [How-to guides](https://docs.nebuly.com/modules/speedster/how-to-guides)
- [Benchmarks](https://docs.nebuly.com/modules/speedster/benchmarks)


# **Key concepts**

`Speedster`'s design reflects the mission to automatically master all the available AI acceleration techniques and deliver the **fastest AI ever.** As a result, `Speedster` leverages available enterprise-grade open-source optimization tools. If these tools and  communities already exist, and are distributed under a permissive license (Apache, MIT, etc), we integrate them and happily contribute to their communities. However, many tools do not exist yet, in which case we implement them and open-source the code so that the community can benefit from it.

`Speedster` is shaped around **4 building blocks** and leverages a modular design to foster scalability and integration of new acceleration components across the stack.

- [x]  **Converter:** converts the input model from its original framework to the framework backends supported by `Speedster`, namely PyTorch, TensorFlow, and ONNX. This allows the Compressor and Optimizer modules to apply any optimization technique to the model.
- [x]  **Compressor:** applies various compression techniques to the model, such as pruning, knowledge distillation, or quantization-aware training.
- [x]  **Optimizer:** converts the compressed models to the intermediate representation (IR) of the supported deep learning compilers. The compilers apply both post-training quantization techniques and graph optimizations, to produce compiled binary files.
- [x]  **Inference Learner:** takes the best performing compiled model and converts it to the same interface as the original input model.

![nebullvm nebuly ai](https://user-images.githubusercontent.com/100476561/180975206-3a3a1f80-afc6-42b0-9953-4b8426c09b62.png)

The **compressor** stage leverages the following open-source projects:

- [Intel/neural-compressor](https://github.com/intel/neural-compressor): targeting to provide unified APIs for network compression technologies, such as low precision quantization, sparsity, pruning, knowledge distillation, across different deep learning frameworks to pursue optimal inference performance.
- [SparseML](https://github.com/neuralmagic/sparseml): libraries for applying sparsification recipes to neural networks with a few lines of code, enabling faster and smaller models.

The **optimizer stage** leverages the following open-source projects:

- [Apache TVM](https://github.com/apache/tvm): open deep learning compiler stack for cpu, gpu and specialized accelerators.
- [BladeDISC](https://github.com/alibaba/BladeDISC): end-to-end Dynamic Shape Compiler project for machine learning workloads.
- [DeepSparse](https://github.com/neuralmagic/deepsparse): neural network inference engine that delivers GPU-class performance for sparsified models on CPUs.
- [OpenVINO](https://github.com/openvinotoolkit/openvino): open-source toolkit for optimizing and deploying AI inference.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): cross-platform, high performance ML inferencing and training accelerator
- [TensorRT](https://github.com/NVIDIA/TensorRT): C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.
- [TFlite](https://github.com/tensorflow/tflite-micro) and [XLA](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla): open-source libraries to accelerate TensorFlow models.



# **Community**

- **[Discord](https://discord.gg/RbeQMu886J)**: best for sharing your projects, hanging out with the community and learning about AI acceleration.
- **[GitHub issues](https://github.com/nebuly-ai/nebullvm/issues)**: ideal for suggesting new acceleration components, requesting new features, and reporting bugs and improvements.

We’re developing `Speedster` together with our community so the best way to get started is to pick a `good-first issue`. Please read our [contribution guidelines](https://docs.nebuly.com/welcome/questions-and-contributions) for a deep dive on how to best contribute to our project!

Don't forget to leave a star ⭐ to support the project and happy acceleration 🚀

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://docs.nebuly.com/welcome/questions-and-contributions">Contribute to the library</a>
</p>


<p align="center">
<a href="https://docs.nebuly.com/modules/speedster/installation">Installation</a> •
<a href="https://docs.nebuly.com/modules/speedster/getting-started">Get started</a> •
<a href="https://github.com/nebuly-ai/nebullvm/tree/main/notebooks">Notebooks</a> •
<a href="https://docs.nebuly.com/modules/speedster/benchmarks">Benchmarks</a>
</p>
