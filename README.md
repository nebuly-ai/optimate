<img  src="https://user-images.githubusercontent.com/59367323/167455174-f3935b7c-c0a9-4fde-8560-97ebd920a3b9.png">

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Contribute to the library</a>
</p>

<p align="center">
<a href="#how-nebullvm-works">How nebullvm works</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">Get started</a>
</p>

  
# Nebullvm

**`nebullvm` speeds up AI inference by 2-30x in just a few lines of code 🚀**

- [How nebullvm works](#how-nebullvm-works)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">Get started</a>
- <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started/pytorch-api">Pytorch</a>, <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started/tensorflow-api">TensorFlow</a>, <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started/hugging-face-api">Hugging Face</a> and <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started/onnx-api">ONNX</a> APIs.


## How nebullvm works

> This open-source library takes your AI model as input and outputs an 
> optimized version that runs 2-30 times faster on your hardware. Nebullvm 
> tests multiple optimization techniques (deep learning compilers, 
> quantization, sparsity, distillation, and more) to identify the optimal way
>  to execute your AI model on your specific hardware.

 `nebullvm`  can speed up your model 2 to 10 times without loss of performance, or up to 30 times if you specify that you are willing to trade off a self-defined amount of accuracy/precision for a super-low latency and a lighter model.

The goal of `nebullvm` is to let any developer benefit from the most advanced inference optimization techniques without having to spend countless hours understanding, installing, testing and debugging these powerful technologies.

Do you want to learn more about how nebullvm optimizes your model? Take a look at the <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">documentation</a>.

<img  src="https://user-images.githubusercontent.com/59367323/167450183-8bd2f76e-3b1d-4de7-b00e-5a755097ed19.png">

### So why nebullvm?

🚀 **Superfast**. nebullvm speeds up the response time of AI models to enable real-time AI applications with reduced computing cost and low power consumption.

☘️ **Easy-to-use**. It takes a few lines of code to install the library and optimize your models.

💻 **Deep learning model agnostic.** `nebullvm` supports all the most popular architectures such as transformers, LSTMs, CNNs and FCNs.

🔥 **Framework agnostic**. `nebullvm` supports the most widely used frameworks and provides as output an optimized version of your model with the same interface. At present, nebullvm supports PyTorch, TensorFlow, Hugging Face and ONNX models.

🤖 **Hardware agnostic**. The library now works on most CPUs and GPUs. If you activate the TVM compiler, nebullvm will also support TPUs and other deep learning-specific ASICs.

✨ **Leveraging the best optimization techniques**. There are many inference optimization techniques such as deep learning compilers, quantization, half-precision or distillation, which are all meant to optimize the way your AI models run on your hardware. It would take developers countless hours to install and test them on every model deployment. `nebullvm` does that for you.

Do you like the concept? Leave a ⭐ if you enjoy the project and [join the Discord community](https://discord.gg/RbeQMu886J) where we chat about `nebullvm` and AI optimization. And happy acceleration 🚀🚀



## Benchmarks

We have tested `nebullvm` on popular AI models and hardware from leading vendors.

The table below shows the inference speedup provided by `nebullvm`. The speedup is calculated as the response time of the unoptimized model divided by the response time of the accelerated model, as an average over 100 experiments. As an example, if the response time of an unoptimized model was on average 600 milliseconds and after `nebullvm` optimization only 240 milliseconds, the resulting speedup is 2.5x times, meaning 150% faster inference.

A complete overview of the experiment and findings can be found on <a href="https://app.gitbook.com/s/Ofr7q1XcUfo7iYMV6A0C/nebullvm/how-nebullvm-works/optimization-workflow">in the documentation</a>.

|                         |  **M1 Pro**  | **Intel Xeon** | **AMD EPYC** | **Nvidia T4** |
|-------------------------|:------------:|:---------------:|:-------------:|:-------------:|
|      **EfficientNetB0** |     23.3x    |       3.5x      |      2.7x     |      1.3x     |
|      **EfficientNetB2** |     19.6x    |       2.8x      |      1.5x     |      2.7x     |
|      **EfficientNetB6** |     19.8x    |       2.4x      |      2.5x     |      1.7x     |
|            **Resnet18** |     1.2x     |       1.9x      |      1.7x     |      7.3x     |
|           **Resnet152** |     1.3x     |       2.1x      |      1.5x     |      2.5x     |
|          **SqueezeNet** |     1.9x     |       2.7x      |      2.0x     |      1.3x     |
|       **Convnext tiny** |     3.2x     |       1.3x      |      1.8x     |      5.0x     |
|      **Convnext large** |     3.2x     |       1.1x      |      1.6x     |      4.6x     |
|    **GPT2 - 10 tokens** |     2.8x     |       3.2x      |      2.8x     |      3.8x     |
|  **GPT2 - 1024 tokens** |       -      |       1.7x      |      1.9x     |      1.4x     |
|     **Bert - 8 tokens** |     6.4x     |       2.9x      |      4.8x     |      4.1x     |
|   **Bert - 512 tokens** |     1.8x     |       1.3x      |      1.6x     |      3.1x     |
| ____________________ | ____________ |   ____________  |  ____________ |  ____________ |

 
Overall, the library provides great results, with more than 2x acceleration in most cases and around 20x in a few applications. We can also observe that acceleration varies greatly across different hardware-model couplings, so we suggest you test `nebullvm` on your model and hardware to assess its full potential on your specific use case.

Besides, across all scenarios, `nebullvm` is very helpful for its ease of use, allowing you to take advantage of inference optimization techniques without having to spend hours studying, testing and debugging these powerful technologies.

## Installation

The installation consists of two steps:
- [`nebullvm` installation](#nebullvm-installation)
- [Installation of deep learning compilers](#installation-of-deep-learning-compilers)

### Nebullvm installation

There are two ways to install `nebullvm`:
- [Using PyPI](#installation-with-pypi-recommended). We suggest installing the library with pip to get the stable version of nebullvm
- [From source code](#installation-from-source-code) to get the latest features

#### Installation with PyPI (recommended)

The easiest way to install `nebullvm` is by using `pip`, running

```
pip install nebullvm
```
#### Installation from source code

Alternatively, you can install nebullvm from source code by cloning the directory on your local machine 
using `git`.

```
git clone https://github.com/nebuly-ai/nebullvm.git
```
Then, enter the repo and install `nebullvm` with `pip`.

```
cd nebullvm
pip install .
```

### Installation of deep learning compilers

Follow the instructions below to automatically install all deep learning compilers leveraged by nebullvm (OpenVINO, TensorRT, ONNX Runtime, Apache TVM, etc.).

To install them, there are thee ways:

- [Installation at the first optimization run](#installation-at-the-first-optimization-run)
- [Installation before the first optimization run (recommended)](#installation-before-the-first-optimization-run-recommended)
- [Download Docker images with preinstalled compilers](#download-docker-images-with-preinstalled-compilers)

Note that:
- Apache TVM is not installed with the below instructions. TVM can be installed separately by following this [guide](https://nebuly.gitbook.io/nebuly/nebullvm/installation/install-and-activate-the-apache-tvm-compiler).
- As an alternative to automatic installation of all compilers, they can be selectively installed by following these [instructions](https://nebuly.gitbook.io/nebuly/nebullvm/installation/selective-installation-of-deep-learning-compilers).

#### Installation at the first optimization run

The automatic installation of the deep learning compilers is activated after you `import nebullvm` and perform your first optimization. You may run into import errors related to the deep learning compiler installation, but you can ignore these errors/warnings. It is also recommended re-starting the python kernel between the auto-installation and the first optimization, otherwise not all compilers will be activated.

#### Installation before the first optimization run (recommended)

To avoid any problems, we strongly recommend running the auto-installation 
before performing the first optimization by running

```
python -c "import nebullvm"
```

You should ignore at this stage any import warning resulting from the previous 
command.

#### Download Docker images with preinstalled compilers
Instead of installing the compilers, which may take a long time, you can simply download the docker container with all compilers preinstalled and start using nebullvm.
To pull the docker image you can simply run

```
docker pull nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-allcompilers
```

and you can then run and access the docker with

```
docker run -ia nebulydocker/nebullvm:cuda11.2.0-nebullvm0.3.1-allcompilers
```

After you have compiled the model, you may decide to deploy it to production. Note that some of the components used to optimize the model are also needed to run it, so **you must have the compiler installed in the production docker**. For this reason, we have created several versions of our Docker container in the [Docker Hub](https://hub.docker.com/repository/docker/nebulydocker/nebullvm), each containing only one compiler. Pull the image with the compiler that has optimized your model!



## Get started, APIs and tutorials
`nebullvm` reduces the computation time of deep learning model inference by 2-30 times by testing multiple optimization techniques and identifying the optimal way to execute your AI model on your hardware.

`nebullvm` can be deployed in two ways:
- [Option A: 2-10x acceleration, NO performance loss](#option-a-2-10x-acceleration-no-performance-loss)
- [Option B: 2-30x acceleration, supervised performance loss](#option-b-2-30x-acceleration-supervised-performance-loss)

For a detailed explanation of how nebullvm works and how to use it, refer to the <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">documentation</a>.

### Option A: 2-10x acceleration, NO performance loss

If you choose this option, `nebullvm` will test multiple deep learning compilers (TensorRT, OpenVINO, ONNX Runtime, etc.) and identify the optimal way to compile your model on your hardware, increasing inference speed by 2-10 times without affecting the performance of your model.

As an example, below is code for accelerating a PyTorch model with nebullvm's PyTorch API.

```
>>> import torch
>>> import torchvision.models as models
>>> from nebullvm import optimize_torch_model
>>> model = models.efficientnet_b0()
>>> save_dir = "."
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> optimized_model = optimize_torch_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>> x = torch.randn(1, 3, 256, 256)
>>> res = optimized_model(x)
```

### Option B: 2-30x acceleration, supervised performance loss

`Nebullvm` is capable of speeding up inference by much more than 10 times in case you are willing to sacrifice a fraction of your model's performance. If you specify how much performance loss you are willing to sustain, `nebullvm` will push your model's response time to its limits by identifying the best possible blend of state-of-the-art inference optimization techniques, such as deep learning compilers, distillation, quantization, half-precision, sparsity, etc.

<br>

Check out the <a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">documentation</a> for more information on nebullvm APIs, how to use them, and for tutorials. Also find more information on how to [contribute to the library](https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions) and [share feedback](https://nebuly.gitbook.io/nebuly/nebullvm/how-nebullvm-works/continuous-improvement#sharing-feedback-to-improve-nebullvm) to support its continuous improvement.

And leave a star ⭐ to support the project 💫

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://nebuly.gitbook.io/nebuly/welcome/questions-and-contributions">Contribute to the library</a>
</p>

<p align="center">
<a href="#how-nebullvm-works">How nebullvm works</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="https://nebuly.gitbook.io/nebuly/nebullvm/get-started">Get started</a>
</p>
