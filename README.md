<img  src="https://user-images.githubusercontent.com/59367323/167455174-f3935b7c-c0a9-4fde-8560-97ebd920a3b9.png">

<p align="center">
<a href="https://github.com/nebuly-ai/nebullvm/releases">NEW RELEASE</a>
</p>

<p align="center">
<a href="#how-nebullvm-works">How Nebullvm Works</a> •
<a href="#tutorials">Tutorials</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="#get-started">Get Started</a> •
<a href="#optimization-examples">Optimization Examples</a>
</p>

<p align="center">
<a href="https://discord.gg/RbeQMu886J">Discord</a> |
<a href="https://nebuly.ai/](https://nebuly.ai/">Website</a> |
<a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
<a href="https://twitter.com/nebuly_ai">Twitter</a>
</p>
  
  # Nebullvm

**`nebullvm` speeds up AI inference by 2-30x in just a few lines of code 🚀**

-  [How Nebullvm Works](#how-nebullvm-works)
-  [Benchmarks](#benchmarks)
-  [Installation and Get Started](#installation)
-  [Optimization Examples](#optimization-examples)
-  [Tutorials](#tutorials)
-  <a href="https://discord.gg/jyjtZTPyHS">Join the Community for AI Acceleration</a>



## How Nebullvm Works

> This open-source library takes your AI model as input and outputs an 
> optimized version that runs 2-30 times faster on your hardware. Nebullvm 
> tests multiple optimization techniques (deep learning compilers, 
> quantization, sparsity, distillation, and more) to identify the optimal way
>  to execute your AI model on your specific hardware. The library can speed 
>  up your model 2 to 10 times without loss of performance, or up to 30 times 
>  if you specify that you are willing to trade off a self-defined amount of 
>  accuracy/precision for a super-low latency and a lighter model.


The goal of `nebullvm` is to let any developer benefit from the most advanced inference optimization techniques without having to spend countless hours understanding, installing, testing and debugging these powerful technologies.


The library aims to be:

☘️ **Easy-to-use**. It takes a few lines of code to install the library and optimize your models.

🔥 **Framework agnostic**. `nebullvm` supports the most widely used frameworks (PyTorch, TensorFlow, ONNX and Hugging Face, etc.) and provides as output an optimized version of your model with the same interface (PyTorch, TensorFlow, etc.).

💻 **Deep learning model agnostic.** `nebullvm` supports all the most popular architectures such as transformers, LSTMs, CNNs and FCNs.

🤖 **Hardware agnostic**. The library now works on most CPUs and GPUs and will soon support TPUs and other deep learning-specific ASICs.

🔑 **Secure.** Everything runs locally on your machine.

✨ **Leveraging the best optimization techniques**. There are many inference optimization techniques such as deep learning compilers, quantization, half precision or distillation, which are all meant to optimize the way your AI models run on your hardware. It would take a developer countless hours to install and test them on every model deployment. The library does that for you.

Do you like the concept? Leave a ⭐ if you enjoy the project and [join the Discord community](https://discord.gg/RbeQMu886J) where we chat about `nebullvm` and AI optimization. And happy acceleration 🚀🚀


<img  src="https://user-images.githubusercontent.com/59367323/167450183-8bd2f76e-3b1d-4de7-b00e-5a755097ed19.png">

## Benchmarks

We have tested `nebullvm` on popular AI models and hardware from leading vendors.

The table below shows the inference speedup provided by `nebullvm`. The speedup is calculated as the response time of the unoptimized model divided by the response time of the accelerated model, as an average over 100 experiments. As an example, if the response time of an unoptimized model was on average 600 milliseconds and after `nebullvm` optimization only 240 milliseconds, the resulting speedup is 2.5x times, meaning 150% faster inference.

A complete overview of the experiment and findings can be found on <a href="https://github.com/nebuly-ai/nebullvm/resources/README.md">this page</a>.

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

 
Overall, the library provides great results, with more than 2x acceleration in most cases and around 20x in a few applications. We can also observe that acceleration varies greatly across different hardware-model couplings, so we suggest you test `nebullvm` on your model and hardware to assess its full potential. You can find the instructions below.

Besides, across all scenarios, `nebullvm` is very helpful for its ease of use, allowing you to take advantage of inference optimization techniques without having to spend hours studying, testing and debugging these technologies.


## Tutorials

We suggest testing the library on your AI models right away by following the installation instructions below. If you want to get a first feel of the library's capabilities or take a look at how `nebullvm` can be readily implemented in an AI workflow, we have built 3 tutorials and notebooks where the library can be tested on the most popular AI frameworks TensorFlow, PyTorch and Hugging Face.

- <a href="https://github.com/nebuly-ai/nebullvm/resources/notebooks/Accelerate-FastAI-Resnet34-with-nebullvm.ipynb">Notebook</a>: Accelerate FastAI's Resnet34 with nebullvm
- <a href="https://github.com/nebuly-ai/nebullvm/resources/notebooks/Accelerate-PyTorch-YOLO-with-nebullvm.ipynb">Notebook</a>: Accelerate PyTorch YOLO with nebullvm
- <a href="https://github.com/nebuly-ai/nebullvm/resources/notebooks/Accelerate-Hugging-Face-GPT2-and-BERT-with-nebullvm.ipynb">Notebook</a>: Accelerate Hugging Face's GPT2 and BERT with nebullvm

## Installation

<details> 
<summary> Step 1: Installation of nebullvm library </summary>


There are two ways to install `nebullvm`:

1. Using PyPI. We suggest installing the library with `pip` to get the stable 
version of `nebullvm`
2. From source code to get the latest features

#### Option 1A: Installation with PyPI (recommended)

The easiest way to install `nebullvm` is by using `pip`, running

```
pip install nebullvm
```

#### Option 1B: Source code installation

To install the source code you have to clone the directory on your local machine 
using `git`.

```
git clone https://github.com/nebuly-ai/nebullvm.git
```
Then, enter the repo and install `nebullvm` with `pip`.

```
cd nebullvm
pip install .
```

</details>

  
<details> 
<summary> Step 2: Installation of deep learning compilers </summary>

Now you need to install the compilers that the library leverages to create the 
optimized version of your models. We have built an auto-installer to install them 
automatically.

#### Option 2A: Installation at the first optimization run

The auto-installer is activated after you import `nebullvm` and perform your 
first optimization. You may run into import errors related to the deep learning 
compiler installation, but you can ignore these errors/warnings. 
It is also recommended restarting the python kernel between the auto-installation 
and the first optimization, otherwise not all compilers will be activated.

#### Option B: Installation before the first optimization run (recommended)

To avoid any problems, we strongly recommend running the auto-installation 
before performing the first optimization by running

```
python -c "import nebullvm"
```

You should ignore at this stage any import warning resulting from the previous 
command.

#### Option 2C: Selective installation of deep learning compilers

The library automatically installs all deep learning compilers it supports. In case you would be interested in bypassing the automatic installation, you can export the environment variable 
`NO_COMPILER_INSTALLATION=1` by running

```
export NO_COMPILER_INSTALLATION=1
```

from your command line or adding

```
import os
os.environ["NO_COMPILER_INSTALLATION"] = "1"
```

in your python code before importing `nebullvm` for the first time.

Note that auto-installation of open-source compilers is done outside the 
`nebullvm` wheel. Installations of ApacheTVM and Openvino have been tested 
on macOS, linux distributions similar to Debian and CentOS.

The feature is still in an alpha version, so we expect that it may fail under 
untested circumstances.

### Step 2-bis: Install TVM
Since the TVM compiler has to be installed from source code, its installation can take several minutes, or even hours, to complete. For this reason, we decided not to include it in the default automatic installer. However, if you want to squeeze the most performance out of your model on your machine, we highly recommend installing TVM as well. With nebullvm, installing TVM becomes very easy, just run
```
python -c "from nebullvm.installers.installers import install_tvm; install_tvm()"
```
and wait for the compiler to be installed! You can check that everything worked 
running
```
python -c "from tvm.runtime import Module"
```

</details>

<details> 
<summary> Possible installation issues </summary>

**MacOS**: the installation may fail on MacOS for MacBooks with the Apple Silicon 
chip, due to scipy compilation errors. The easy fix is to install `scipy` with 
another package manager such as conda (the Apple Silicon distribution of 
Mini-conda) and then install `nebullvm`. For any additional issues do not 
hesitate to open an issue or contact directly `info@nebuly.ai` by email.

</details>



## Get Started
`Nebullvm` reduces the computation time of deep learning model inference by 2-30 times by testing multiple optimization techniques (deep learning compilers, quantization, half precision, distillation, and more) and identifying the optimal way to execute your AI model on your specific hardware.

`Nebullvm` can be deployed in two ways.

### Option A: 2-10x acceleration, NO performance loss

If you choose this option, `nebullvm` will test multiple deep learning compilers (TensorRT, OpenVINO, ONNX Runtime, etc.) and identify the optimal way to compile your model on your hardware, increasing inference speed by 2-10 times without affecting the performance of your model.

### Option B: 2-30x acceleration, supervised performance loss

`Nebullvm` is capable of speeding up inference by much more than 10 times in case you are willing to sacrifice a fraction of your model's performance. If you specify how much performance loss you are willing to sustain, `nebullvm` will push your model's response time to its limits by identifying the best possible blend of state-of-the-art inference optimization techniques, such as deep learning compilers, distillation, quantization, half precision, sparsity, etc.

Performance monitoring is accomplished using the `perf_loss_ths` (performance loss threshold), and the `perf_metric` for performance estimation.

#### Option B.1 
  
When a predefined metric (e.g. “accuracy”) or a custom metric is passed as the `perf_metric` argument, the value of `perf_loss_ths` will be used as the maximum acceptable loss for the given metric evaluated on your datasets.

#### Options B.2 and B.3
When no `perf_metric` is provided as input, `nebullvm` calculates the performance loss using the default `precision` function. If the `dataset` is provided, the precision will be calculated on 100 sampled data (option B.2). Otherwise, the data will be randomly generated from the metadata provided as input, i.e. `input_sizes` and `batch_size` (option B.3).

<details> 
<summary> Options B.2 and B.3: Impact of perf_loss_ths on precision </summary>

The table below shows the impact of `perf_loss_ths` on the default metric `"precision"`.

| **perf_loss_ths** | **Expected behavior with the default “precision” metric**                                                                                                                                                                                  |
|:-----------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **None or 0**     | No precision-reduction technique (distillation, quantization, half precision, sparsity, etc.) will be applied, as per Option A.                                                                                                            |
| **1**             | Nebullvm will accept the outcome of precision-reduction techniques only if the relative change of the smallest output logit is smaller than 1. This is usually correlated with a marginal drop in precision.                               |
| **2**             | Nebullvm will accept a "riskier" output from precision-reduction techniques to achieve increased inference speed. This can usually have an impact on the accuracy of ~0.1%.                                                                 |
| **≥3**            | Aggressive precision reduction techniques are used to produce the lightest and fastest model possible. Accuracy drops depend on both model type and task type. A simple binary classification can still show accuracy drops around ~0.1%.  | 

</details>

<img  src="https://user-images.githubusercontent.com/83510798/167514430-896577a1-7d70-416a-b170-5d861ba58cad.png">


## Optimization examples
<details> 
<summary> Optimization with PyTorch </summary>
Here we present an example of optimizing a `pytorch` model with `nebullvm`:

```
>>> # FOR EACH OPTION
>>> import torch
>>> import torchvision.models as models
>>> from nebullvm import optimize_torch_model
>>> model = models.efficientnet_b0()
>>> save_dir = "."
>>>
>>> # ONLY FOR OPTION A 
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> optimized_model = optimize_torch_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>>
>>> # ONLY FOR OPTION B.1
>>> dl = [((torch.randn(1, 3, 256, 256), ), 0)]
>>> perf_loss_ths = 0.1  # We can accept a drop in the loss function up to 10%
>>> optimized_model = optimize_torch_model(
... model, dataloader=dl, save_dir=save_dir, perf_loss_ths=perf_loss_ths, perf_metric="accuracy", 
... )
>>>
>>> # ONLY FOR OPTION B.2
>>> dl = [((torch.randn(1, 3, 256, 256), ), 0)]
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> optimized_model = optimize_torch_model(
... model, dataloader=dl, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # ONLY FOR OPTION B.3
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> optimized_model = optimize_torch_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # FOR EACH OPTION
>>> x = torch.randn(bs, 3, 256, 256)
>>> res = optimized_model(x)
```

In the example above for options B.1 and B.2 we provided a dataset containing a single tuple `(xs, y)` where `xs` itself is a tuple containing all the inputs needed for the model. Note that for `nebullvm` input dataset should be in the format `Sequence[Tuple[Tuple[Tensor, ...], TensorOrNone]]`. The torch API also accept `dataloaders` as inputs, however the `dataloader` should return each batch as a tuple `(xs, y)` as described before.

</details> 

<details> 
<summary> Optimization with TensorFlow </summary>

```
>>> # FOR EACH OPTION
>>> import tensorflow as tf 
>>> from tensorflow.keras.applications.resnet50 import ResNet50
>>> from nebullvm import optimize_tf_model
>>> model = ResNet50()
>>> save_dir = "."
>>>
>>> # ONLY FOR OPTION A
>>> bs, input_sizes = 1, [(224, 224, 3)]
>>> optimized_model = optimize_tf_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>>
>>> # ONLY FOR OPTION B.1
>>> input_data = [((tf.random_normal_inizializer()(shape=(1, 224, 224, 3)), ), 0)]
>>> perf_loss_ths = 0.1  # We can accept a drop in the loss function up to 10%
>>> optimized_model = optimize_tf_model(
... model, dataset=input_data, save_dir=save_dir, perf_loss_ths=perf_loss_ths, perf_metric="accuracy", 
... )
>>>
>>> # ONLY FOR OPTION B.2
>>> input_data = [((tf.random_normal_inizializer()(shape=(1, 224, 224, 3)), ), 0)]
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> optimized_model = optimize_tf_model(
... model, dataset=input_data, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # ONLY FOR OPTION B.3
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> bs, input_sizes = 1, [(224, 224, 3)]
>>> optimized_model = optimize_tf_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # FOR EACH OPTION
>>> res = optimized_model(*optimized_model.get_inputs_example())
```
  
</details>

<details> 
<summary> Optimization with ONNX </summary>

```
>>> # FOR EACH OPTION
>>> from nebullvm import optimize_torch_model
>>> import numpy as np
>>> model_path = "path-to-onnx-model"
>>> save_dir = "."
>>>
>>> # ONLY FOR OPTION A
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> optimized_model = optimize_onnx_model(
... model_path, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>>
>>> # ONLY FOR OPTION B.1
>>> data = [((np.random.randn(1, 3, 256, 256).astype(np.float32), ), 0)]
>>> perf_loss_ths = 0.1  # We can accept a drop in the loss function up to 10%
>>> optimized_model = optimize_onnx_model(
... model_path, data=data, save_dir=save_dir, perf_loss_ths=perf_loss_ths, perf_metric="accuracy", 
... )
>>>
>>> # ONLY FOR OPTION B.2
>>> data = [((np.random.randn(1, 3, 256, 256).astype(np.float32), ), 0)]
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> optimized_model = optimize_onnx_model(
... model_path, data=data, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # ONLY FOR OPTION B.3
>>> perf_loss_ths = 2  # Relative error on the smallest logits accepted
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> optimized_model = optimize_onnx_model(
... model_path, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir, perf_loss_ths=perf_loss_ths, 
... )
>>>
>>> # FOR EACH OPTION
>>> x = np.random.randn(1, 3, 256, 256).astype(np.float32)
>>> res = optimized_model(x)
```

</details>

<details> 
<summary> Optimization with Hugging Face </summary>

To make `nebullvm` work with `huggingface` we have changed the API slightly so that you can use the `optimize_huggingface_model` function to optimize your model. Below we show an example of how to accelerate GPT2 with `nebullvm` without loss of accuracy by leveraging only deep learning compilers (option A).

```
>>> from transformers import GPT2Tokenizer, GPT2Model
>>> from nebullvm.api.frontend.huggingface import optimize_huggingface_model
>>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
>>> model = GPT2Model.from_pretrained('gpt2')
>>> text = "Replace me by any text you'd like."
>>> encoded_input = tokenizer(text, return_tensors='pt')
>>> optimized_model = optimize_huggingface_model(
...     model=model,
...     tokenizer=tokenizer,
...     input_texts=[text],
...     batch_size=1,
...     max_input_sizes=[
...       tuple(value.size()[1:]) 
...       for value in encoded_input.values()
...     ],
...     save_dir=".",
...     extra_input_info=[{}, {"max_value": 1, "min_value": 0}],
... )
>>> res = optimized_model(**encoded_input)
```

</details>


### Set the number of threads per model
When running multiple replicas of the model in parallel, it would be useful 
for CPU-optimized algorithms to limit the number of threads to use for each model.
In nebullvm, it is possible to set the maximum number of threads a single model 
can use with the environment variable `NEBULLVM_THREADS_PER_MODEL`. 
For instance, you can run
```bash
export NEBULLVM_THREADS_PER_MODEL = 2
```
for using just two CPU-threads per model at inference time and during optimization.


## Supported frameworks

- PyTorch
- TensorFlow
- Hugging Face

## Supported deep learning compilers

- OpenVINO
- TensorRT
- TVM
- MLIR (Coming soon 🚀)


## Integration with other open-source libraries

Deep learning libraries

- [Clip-as-Service by Jina AI](https://github.com/jina-ai/clip-as-service) ![](https://img.shields.io/github/stars/jina-ai/clip-as-service.svg?style=social) Embed images and sentences into fixed-length vectors with CLIP. (🚀 coming soon   🌖)
- [SignLangGNN by Anindyadeep](https://github.com/Anindyadeep/SignLangGNN) ![](https://img.shields.io/github/stars/Anindyadeep/SignLangGNN.svg?style=social) ASL sign language classification on static images using graph neural networks.
    
Repositories of the best tools for AI

- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning#tools) ![](https://img.shields.io/github/stars/ChristosChristofidis/awesome-deep-learning.svg?style=social) A curated list of awesome Deep Learning tutorials, projects and communities.
- [Awesome Production ML](https://github.com/EthicalML/awesome-production-machine-learning#optimized-computation-frameworks) ![](https://img.shields.io/github/stars/EthicalML/awesome-production-machine-learning.svg?style=social) A list of open-source libraries to deploy, monitor, version and scale your machine learning.
- [Best of ML Python](https://github.com/ml-tooling/best-of-ml-python#model-serialization--deployment) ![](https://img.shields.io/github/stars/ml-tooling/best-of-ml-python.svg?style=social) - A ranked list of awesome machine learning Python libraries.
- [Awesome MLOps](https://github.com/EthicalML/awesome-production-machine-learning#optimization-tools) ![](https://img.shields.io/github/stars/kelvins/awesome-mlops.svg?style=social) - A curated list of MLOps tools.

Do you want to integrate nebullvm in your open-source library? Try it out and if you need support, do not hesitate to contact us at [info@nebuly.ai](mailto:info@nebuly.ai).




## The community for AI acceleration

Do you want to meet nebullvm contributors and other developers who share the vision of an superfast and sustainable artificial intelligence? Or would you like to report bugs or improvement ideas for nebullvm? [Join the community](https://discord.gg/RbeQMu886J) for AI acceleration on Discord!

## Acknowledgments

`Nebullvm` was built by [Nebuly](https://nebuly.ai/), with a major contribution by [Diego Fiori](https://www.linkedin.com/in/diego-fiori-b64b3016a/), as well as a lot of support from the community who submitted pull requests, provided very useful feedback, and opened issues. 

`Nebullvm` builds on the outstanding work being accomplished by the open-source community and major hardware vendors on deep learning compilers.

- [OpenVINO](https://docs.openvino.ai/latest/index.html) (on [Intel](https://www.intel.com/) machines)
- [TensorRT](https://developer.nvidia.com/tensorrt) (on [NVIDIA](https://www.nvidia.com/) GPUs)
- [Apache TVM](https://tvm.apache.org/)

---


<p align="center">
<a href="#how-nebullvm-works">How Nebullvm Works</a> •
<a href="#tutorials">Tutorials</a> •
<a href="#benchmarks">Benchmarks</a> •
<a href="#installation">Installation</a> •
<a href="#get-started">Get Started</a> •
<a href="#optimization-examples">Optimization Examples</a>
</p>

<p align="center">
<a href="https://discord.gg/RbeQMu886J">Discord</a> |
<a href="https://nebuly.ai/](https://nebuly.ai/">Website</a> |
<a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
<a href="https://twitter.com/nebuly_ai">Twitter</a>
</p>




