
![nebullvm nebuly logo](https://user-images.githubusercontent.com/100476561/180968989-31bebd47-e789-42a5-9a40-71a19c025389.png)


# **Nebullvm**

`nebullvm` is an open-source tool designed to speed up AI inference in just a few lines of code. `nebullvm` boosts your model to achieve the maximum acceleration that is physically possible on your hardware.

We are building a new AI inference acceleration product leveraging state-of-the-art open-source optimization tools enabling the optimization of the whole software to hardware stack. If you like the idea, give us a star to support the project ⭐

![nebullvm nebuly](https://user-images.githubusercontent.com/83510798/187256101-b557f9f8-3681-47b1-bac7-d70189c0e80a.png)


The core `nebullvm` workflow consists of 3 steps:

- [x]  **Select**: input your model in your preferred DL framework and express your preferences regarding:
    - Accuracy loss: do you want to trade off a little accuracy for much higher performance?
    - Optimization time: stellar accelerations can be time-consuming. Can you wait, or do you need an instant answer?
- [x]  **Search**: `nebullvm` automatically tests every combination of optimization techniques across the software-to-hardware stack (sparsity, quantization, compilers, etc.) that is compatible with your needs and local hardware.
- [x]  **Serve**: finally, `nebullvm` chooses the best configuration of optimization techniques and returns an accelerated version of your model in the DL framework of your choice (just on steroids 🚀).

# API quick view

Only a single line of code is needed to get your accelerated model:

```python
import torch
import torchvision.models as models
from nebullvm.api.functions import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [((torch.randn(1, 3, 256, 256), ), 0)]

# Run nebullvm optimization in one line of code
optimized_model = optimize_model(
    model, input_data=input_data, optimization_time="constrained"
)

# Try the optimized model
x = torch.randn(1, 3, 256, 256)
res = optimized_model(x)
```

For more details, please visit [Installation](https://nebuly.gitbook.io/nebuly/nebullvm/installation) and [Get started](https://nebuly.gitbook.io/nebuly/nebullvm/get-started).

# **How it works**

We are not here to reinvent the wheel, but to build an all-in-one open-source product to master all the available AI acceleration techniques and deliver the **fastest AI ever.** As a result, `nebullvm` leverages available enterprise-grade open-source optimization tools. If these tools and  communities already exist, and are distributed under a permissive license (Apache, MIT, etc), we integrate them and happily contribute to their communities. However, many tools do not exist yet, in which case we implement them and open-source the code so that the community can benefit from it.

### **Product design**

`nebullvm` is shaped around **4 building blocks** and leverages a modular design to foster scalability and integration of new acceleration components across the stack.

- [x]  **Converter:** converts the input model from its original framework to the framework backends supported by `nebullvm`, namely PyTorch, TensorFlow, and ONNX. This allows the Compressor and Optimizer modules to apply any optimization technique to the model.
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

- [Installation](https://nebuly.gitbook.io/nebuly/nebullvm/installation)
- [Get started](https://nebuly.gitbook.io/nebuly/nebullvm/get-started)
- [Notebooks](https://nebuly.gitbook.io/nebuly/nebullvm/get-started/notebooks-for-testing-nebullvm)
- [Benchmarks](https://nebuly.gitbook.io/nebuly/nebullvm/benchmarks)
- [Supported features and roadmap](https://nebuly.gitbook.io/nebuly/nebullvm/how-nebullvm-works/supported-features-and-roadmap)

# **Community**

- **[Discord](https://discord.gg/RbeQMu886J)**: best for sharing your projects, hanging out with the community and learning about AI acceleration.
- **[Github issues](https://github.com/nebuly-ai/nebullvm/issues)**: ideal for suggesting new acceleration components, requesting new features, and reporting bugs and improvements.

We’re developing `nebullvm` together with our community so the best way to get started is to pick a `good-first issue`. Please read our [contribution guidelines](https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions) for a deep dive on how to best contribute to our project!

Don't forget to leave a star ⭐ to support the project and happy acceleration 🚀

# **Status**

- **Model converter backends**
    - [x]  ONNX, PyTorch, TensorFlow
    - [ ]  Jax
- **Compressor**
    - [x]  Pruning and sparsity
    - [ ]  Quantized-aware training, distillation, layer replacement and low rank compression
- **Optimizer**
    - [x]  TensorRT, OpenVINO, ONNX Runtime, TVM, PyTorch, DeepSparse, BladeDisc
    - [ ]  TFlite, XLA
- **Inference learners**
    - [x]  PyTorch, ONNX, Hugging Face, TensorFlow
    - [ ]  Jax

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Contribute to the library</a>
</p>


<p align="center">
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/installation">Installation</a> •
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">Get started</a> •
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started/notebooks-for-testing-nebullvm">Notebooks</a> •
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/benchmarks">Benchmarks</a>
</p>
