# 💥 Speedster App

Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware.


## 📖 Description 
`Speedster` is an open-source App designed to speed up AI inference in just a few lines of code. The library boosts your model to achieve the maximum acceleration that is physically possible on your hardware.

We are building a new AI inference acceleration product leveraging state-of-the-art open-source optimization tools enabling the optimization of the whole software to hardware stack. If you like the idea, give us a star to support the project ⭐


![nebullvm benchmarks](https://user-images.githubusercontent.com/28647171/201156822-00a307d7-e3b9-4121-aa83-c7d2a167f791.png)



The core `Speedster` workflow consists of 3 steps:

- [x]  **Select**: input your model in your preferred DL framework and express your preferences regarding:
    - Accuracy loss: do you want to trade off a little accuracy for much higher performance?
    - Optimization time: stellar accelerations can be time-consuming. Can you wait, or do you need an instant answer?
- [x]  **Search**: the App automatically tests every combination of optimization techniques across the software-to-hardware stack (sparsity, quantization, compilers, etc.) that is compatible with your needs and local hardware.
- [x]  **Serve**: finally, `Speedster` chooses the best configuration of optimization techniques and returns an accelerated version of your model in the DL framework of your choice (just on steroids 🚀).


# Installation

> :warning: For **MacOS** with **ARM processors**, please use a conda environment.

Install Speedster and its base requirements:
```
pip install speedster
```

> :warning: If you want to optimize a **PyTorch model**, PyTorch must be pre-installed 
> on your environment before proceeding to the next step, please install it from this 
> [link](https://pytorch.org/get-started/locally/).


Install the deep learning compilers:
```
python -m nebullvm.installers.auto_installer \
    --backends all --compilers all
```

For more details on the installation step, please visit [Installation](https://docs.nebuly.com/speedster/installation).


# API quick view

Only a single line of code is needed to get your accelerated model:

```python
from speedster import optimize_model

optimized_model = optimize_model(model, input_data=input_data)
```
Checkout how to define the `model` and `input_data` parameters depending on which framework you want to use and how to use the optimized model: 
[PyTorch](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks/speedster/pytorch#pytorch-api-quick-view), 
[HuggingFace](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks/speedster/huggingface#huggingface-api-quick-view), 
[TensorFlow](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks/speedster/tensorflow#tensorflow-api-quick-view), 
[ONNX](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks/speedster/onnx#onnx-api-quick-view).

For more details, please visit also the documentation sections [Get Started](https://docs.nebuly.com/speedster/get-started),  [Speedster API](https://docs.nebuly.com/speedster/get-started/speedster-api) and [Examples of API options](https://docs.nebuly.com/speedster/get-started/examples-of-api-options).

# **How it works**

We are not here to reinvent the wheel, but to build an all-in-one open-source product to master all the available AI acceleration techniques and deliver the **fastest AI ever.** As a result, `Speedster` leverages available enterprise-grade open-source optimization tools. If these tools and  communities already exist, and are distributed under a permissive license (Apache, MIT, etc), we integrate them and happily contribute to their communities. However, many tools do not exist yet, in which case we implement them and open-source the code so that the community can benefit from it.

### **Product design**

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

# **Documentation**

- [Installation](https://docs.nebuly.com/speedster/installation)
- [Get started](https://docs.nebuly.com/speedster/get-started)
- [Notebooks](https://github.com/nebuly-ai/nebullvm/tree/main/notebooks)
- [Benchmarks](https://docs.nebuly.com/speedster/benchmarks)
- [Supported features and roadmap](https://docs.nebuly.com/speedster/how-speedster-works/supported-features-and-roadmap)

# **Community**

- **[Discord](https://discord.gg/RbeQMu886J)**: best for sharing your projects, hanging out with the community and learning about AI acceleration.
- **[GitHub issues](https://github.com/nebuly-ai/nebullvm/issues)**: ideal for suggesting new acceleration components, requesting new features, and reporting bugs and improvements.

We’re developing `Speedster` together with our community so the best way to get started is to pick a `good-first issue`. Please read our [contribution guidelines](https://docs.nebuly.com/welcome/questions-and-contributions) for a deep dive on how to best contribute to our project!

Don't forget to leave a star ⭐ to support the project and happy acceleration 🚀

# **Status**

- **Model converter backends**
    - [x]  ONNX, PyTorch, TensorFlow
    - [ ]  Jax
- **Compressor**
    - [x]  Pruning and sparsity
    - [ ]  Quantized-aware training, distillation, layer replacement and low rank compression
- **Optimizer**
    - [x]  TensorRT, OpenVINO, ONNX Runtime, TVM, PyTorch, DeepSparse, BladeDisc, TFlite, XLA
- **Inference learners**
    - [x]  PyTorch, ONNX, Hugging Face, TensorFlow
    - [ ]  Jax

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://docs.nebuly.com/welcome/questions-and-contributions">Contribute to the library</a>
</p>


<p align="center">
<a href="https://docs.nebuly.com/speedster/installation">Installation</a> •
<a href="https://docs.nebuly.com/speedster/get-started">Get started</a> •
<a href="https://github.com/nebuly-ai/nebullvm/tree/main/notebooks">Notebooks</a> •
<a href="https://docs.nebuly.com/speedster/benchmarks">Benchmarks</a>
</p>
