<img  src="https://user-images.githubusercontent.com/83510798/155317935-523dcf79-9adb-4131-9511-8e269a1f97dd.png">

<p align="center">
  <a href="#how-nebullvm-works">How Nebullvm Works</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#installation-and-get-started">Installation</a> •
  <a href="#get-started">Get Started</a>
</p>
<p align="center">
  <a href="https://nebuly.ai/">Website</a> |
  <a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
  <a href="https://twitter.com/nebuly_ai">Twitter</a>
  <a href="https://discord.gg/jyjtZTPyHS">Discord</a>
</p>
  
  
  # Nebullvm

**`nebullvm` speeds up AI inference by 5-20x in just a few lines of code 🚀**

-  [How Nebullvm Works](#how-nebullvm-works)
-  [Technology Demonstration and Benchmarks](#technology-demonstration-and-benchmarks)
-  [Installation and Get Started](#installation-and-get-started)

## How Nebullvm Works

> This open-source library takes your AI model as input and outputs an optimized 
> version that runs 5-20 times faster on your machine. Nebullvm tests multiple 
> deep learning compilers to identify the best possible way to execute your 
> model on your specific hardware, without impacting the accuracy of your model.

The goal of `nebullvm` is to let any developer benefit from deep learning (DL) 
compilers without having to spend tons of hours understanding, installing, 
testing and debugging this powerful technology.

#### The library aims to be:

💻  **Deep learning model agnostic.** `nebullvm` supports all the most popular 
architectures such as transformers, LSTMs, CNNs and FCNs.

🤖  **Hardware agnostic**. The library now works on most CPUs and GPUs and 
will soon support TPUs and other deep learning-specific ASICs.

🔥 **Framework agnostic**. `nebullvm` supports the most widely used frameworks 
(PyTorch, TensorFlow and Hugging Face) and will soon support many more.

🔑 **Secure.** Everything runs locally on your machine.

☘️ **Easy-to-use**. It takes a few lines of code to install the library and 
optimize your models.

✨ **Leveraging the best deep learning compilers**. There are tons of DL compilers 
that optimize the way your AI models run on your hardware. It would take tons of 
hours for a developer to install and test them at every model deployment. 
The library does it for you!

Do you like the concept? Leave a ⭐ if you enjoy the project. 
And happy acceleration 🚀🚀

<img  src="https://user-images.githubusercontent.com/83510798/159326080-05e5b0f2-5197-4e0d-8b2a-1185ffeca95e.png">

## Technology demonstration and benchmarks

### Technology demonstration

We suggest testing the library on your AI models right away by following the 
[installation instructions](#installation-and-get-started) below. 
If you want to get a first feel for the library's capabilities, 
we have built 3 notebooks where the library can be tested on the most popular 
AI frameworks Tensorflow, PyTorch and Hugging Face.

-  [Resnet34 with FastAI](https://nebullvm-notebooks.s3.amazonaws.com/Accelerate-FastAI-inference.ipynb)
-  [YOLO with PyTorch](https://nebullvm-notebooks.s3.amazonaws.com/YOLO-Darknet+Optimization.ipynb)
-  [GPT2 and BERT with Hugging Face](https://nebullvm-notebooks.s3.amazonaws.com/Huggingface-Transformers.ipynb)

The notebooks will run locally on your hardware, so you can get an idea of the 
performance you would achieve with `nebullvm` on your AI models. 
Note that it will take several minutes to install the library the first time.

### Benchmarks

We have tested `nebullvm` on popular AI models and hardware from leading vendors.

-   Hardware: M1 Pro, NVIDIA T4, Intel Xeon, AMD EPYC
-   AI Models: EfficientNet, Resnet, SqueezeNet, BERT, GPT2

The table below shows the response time in milliseconds (ms) of the non-optimized 
model and the optimized model for the various model-hardware couplings as an 
average value over 100 experiments. It also displays the **speedup** provided 
by `nebullvm`, where speedup is defined as the response time of the optimized 
model over the response time of the non-optimized model.

|           |M1 Pro|M1 Pro optimized|M1 pro speedup|Intel Xeon|Intel Xeon optimized|Intel Xeon speedup|AMD EPYC |AMD EPYC optimized|AMD EPYC speedup|Nvidia T4|Nvidia T4 optimized|Nvidia T4 speedup|
|---------------------------------|--------------------------------|--------------------------------|--------------|----------|--------------------|------------------|---------|------------------|----------------|---------|-------------------|-----------------|
|EfficientNetB0   |215.0 ms|24.4 ms         |8.8x          |53.6 ms   |19.2 ms             |2.8x              |121.3 ms |47.1 ms           |2.6x            |12.9 ms  |0.3 ms             |39.2x            |
|EfficientNetB1   |278.8 ms|33.6 ms         |8.3x          |74.8 ms   |27.1 ms             |2.8x              |175.0 ms |70.4 ms           |2.5x            |18.0 ms  |0.3 ms             |54.5x            |
|EfficientNetB2   |284.9 ms|36.8 ms         |7.8x          |86.4 ms   |30.0 ms             |2.9x              |199.1 ms |75.1 ms           |2.7x            |36.9 ms  |0.4 ms             |105.5x           |
|EfficientNetB3   |370.1 ms|50.4 ms         |7.4x          |101.8 ms  |42.8 ms             |2.4x              |279.8 ms |118.0 ms          |2.4x            |20.3 ms  |0.3 ms             |59.6x            |
|EfficientNetB4   |558.9 ms|71.0 ms         |7.9x          |136.6 ms  |64.3 ms             |2.1x              |400.5 ms |159.1 ms          |2.5x            |24.9 ms  |0.3 ms             |73.2x            |
|EfficientNetB5   |704.3 ms|99.8 ms         |7.1x          |189.5 ms  |88.9 ms             |2.1x              |570.2 ms |249.5 ms          |2.3x            |31.2 ms  |0.3 ms             |91.9x            |
|Resnet18         |18.5 ms |15.8 ms         |1.2x          |57.4 ms   |37.9 ms             |1.5x              |164.3 ms |121.9 ms          |1.4x            |9.4 ms   |0.3 ms             |27.6x            |
|SqueezeNet       |15.3 ms |7.9 ms          |1.9x          |39.1 ms   |17.3 ms             |2.3x              |119.0 ms |58.7 ms           |2.0x            |8.9 ms   |0.3 ms             |26.1x            |
|GPT2 - 10 tokens |29.7 ms |10.8 ms         |2.8x          |63.4 ms   |44.6 ms             |1.4x              |180.7 ms |59.1 ms           |3.1x            |15.3 ms  |4.4 ms             |3.5x             |
|Bert - 8 tokens  |39.4 ms |6.2 ms          |6.4x          |44.9 ms   |39.3 ms             |1.1x              |148.4 ms |46.5 ms           |3.2x            |10.4 ms  |3.8 ms             |2.7x             |
|Bert - 512 tokens|489.5 ms|276.4 ms        |1.8x          |801.7 ms  |782.8 ms            |1.0x              |5416.7 ms|2710.7 ms         |2.0x            |31.3 ms  |27.4 ms            |1.1x             |
|_________________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|_____________|
 
At first glance, we can observe that speedup varies greatly across hardware-model 
couplings. Overall, the library provides great positive results, most 
ranging from 2 to 30+ times speedup.

To summarize, the results are:

- `nebullvm` provides positive acceleration to non-optimized AI models
- These early results show poorer (yet positive) performance on Hugging Face 
models. Support for Hugging Face has just been released and improvements will be 
included in future versions
- The library provides a ~2-3x boost on Intel and AMD hardware. These results are 
most likely related to an already highly optimized implementation of PyTorch for 
x86 devices
- Nebullvm delivers extremely good performance on NVIDIA machines
- The library provides great performances also on Apple M1 chips

And across all scenarios, `nebullvm` is very useful for its ease of use, 
allowing you to take advantage of deep learning compilers without having 
to spend hours studying, testing and debugging this technology.


## Installation and Get Started

### Step 1: Installation of nebullvm library

There are two ways to install `nebullvm`:

1. Using PyPI. We suggest installing the library with `pip` to get the stable 
version of `nebullvm`
2. From source code to get the latest features

#### Option A: Installation with PyPI (recommended)

The easiest way to install `nebullvm` is by using `pip`, running

```
pip install nebullvm
```

#### Option B: Source code installation

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

### Step 2: Installation of deep learning compilers

Now you need to install the compilers that the library leverages to create the 
optimized version of your models. We have built an auto-installer to install them 
automatically.
Note that it will take several minutes to install all compilers and we recommend 
following the second option below to avoid any installation issues.

#### Option A: Installation at the first optimization run

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

#### Option C: Selective installation of DL compilers

The library automatically installs all DL compilers it supports. However, for 
some reason you may be interested in bypassing the automatic installation. 
If this is the case, you can simply export the environment variable 
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
Since the TVM compiler needs to be installed from the source code, its installation
can take several minutes (or even hours) for being performed. For this reason we
decided to not include it in the default auto-installer. However, if you want to 
squeeze out the maximum of the performance from your model on your machine, we 
highly recommend installing TVM as well. With `nebullvm` it is super-easy! Just run
```
python -c "from nebullvm.installers.installers import install_tvm; install_tvm()"
```
and wait for the compiler to be installed! You can check that everything worked 
running
```
python -c "from tvm.runtime import Module"
```

### Possible installation issues

**MacOS**: the installation may fail on MacOS for MacBooks with the Apple Silicon 
chip, due to scipy compilation errors. The easy fix is to install `scipy` with 
another package manager such as conda (the Apple Silicon distribution of 
Mini-conda) and then install `nebullvm`. For any additional issues do not 
hesitate to open an issue or contact directly `info@nebuly.ai` by email.


## Get Started

`Nebullvm` reduces the computation time of deep learning model inference by 
5-20 times by testing multiple deep learning compilers and identifying the 
best possible way to execute your model on your specific hardware, without 
impacting the accuracy of your model.

Currently `nebullvm` supports models in the `pytorch`, `tensorflow` and 
`huggingface` frameworks, and many others will be included soon. Models can be 
easily imported from one of the supported frameworks using the appropriate 
feature as explained below.

And please leave a ⭐. If many will like the library, we will keep building new 
and cool features. We have a long list of them!

### Optimization with PyTorch

Here we present an example of optimizing a `pytorch` model with `nebullvm`:

```
>>> import torch
>>> import torchvision.models as models
>>> from nebullvm import optimize_torch_model
>>> model = models.efficientnet_b0()
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> save_dir = "."
>>> optimized_model = optimize_torch_model(
... model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>> x = torch.randn((bs, *input_sizes[0]))
>>> res = optimized_model(x)
```

### Optimization with TensorFlow

The same optimization can be achieved with a `tensorflow` model using the 
function `nebullvm.optimize_tf_model`.

```
>>> from nebullvm import optimize_tf_model
>>> from tensorflow.keras.applications.resnet50 import ResNet50
>>> model = ResNet50()
>>> bs, input_sizes = 1, [(224, 224, 3)]
>>> save_dir = "."
>>> optimized_model = optimize_tf_model(
...    model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>> res = optimized_model(*optimized_model.get_inputs_example())
```

### Optimization with ONNX

The similar optimization can be achieved with an `onnx` model using the 
function `nebullvm.optimize_onnx_model`.

```
>>> from nebullvm import optimize_onnx_model
>>> model_path = "path-to-onnx-model"
>>> bs, input_sizes = 1, [(224, 224, 3)]
>>> save_dir = "."
>>> optimized_model = optimize_onnx_model(
...    model_path, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>> res = optimized_model(*optimized_model.get_inputs_example())
```

### Optimization with Hugging Face
To make `nebullvm` work with `huggingface` we changed the API slightly so that 
you can use the `optimize_huggingface_model` function to optimize your model.

Note that the current version of `nebullvm` only supports Hugging Face models 
built on top of `pytorch`. Support for TensorFlow will be included in future 
releases.

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
...     target_text=text,
...     batch_size=1,
...     max_input_sizes=[
...       tuple(value.size()[1:]) 
...       for value in encoded_input.values()
...     ],
...     save_dir=".",
...     extra_input_info=[{}, {"max_value": 1, "min_value": 0}],
...     use_torch_api=False
... )
>>> res = optimized_model(**encoded_input)
```

## Testing the library on your models

If you want to compare the performance of a model optimized by `nebullvm` with 
its non-optimized version, you can find guidelines in the notebooks presented 
in the section <a href="#technology-demonstration">Technology demonstration</a>.

## Supported frameworks

- PyTorch
- TensorFlow
- Hugging Face

## Supported deep learning compilers

- OpenVINO
- TensorRT
- TVM
- MLIR (Coming soon 🚀)

## Community
You are interested in making AI more efficient? You want to meet other people 
sharing the vision of an efficient AI which is actually easy to use without 
needing deep knowledge on the hardware side? Join us in the [Nebuly tribe](https://discord.gg/jyjtZTPyHS) on
Discord!

## Acknowledgments

`Nebullvm` builds on the outstanding work being accomplished by the open-source 
community and major hardware vendors on deep learning compilers. 
Currently `nebullvm` supports as AI compilers:

- [OpenVINO](https://docs.openvino.ai/latest/index.html) (on [Intel](https://www.intel.com/) Machines)
- [TensorRT](https://developer.nvidia.com/tensorrt) (on [NVIDIA](https://www.nvidia.com/) GPUs)
- [Apache TVM](https://tvm.apache.org/)

---

<p align="center">
  <a href="#how-nebullvm-works">How Nebullvm Works</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#installation-and-get-started">Installation</a> •
  <a href="#get-started">Get Started</a>
</p>
<p align="center">
  <a href="https://nebuly.ai/">Website</a> |
  <a href="https://www.linkedin.com/company/72460022/">LinkedIn</a> |
  <a href="https://twitter.com/nebuly_ai">Twitter</a>
  <a href="https://discord.gg/jyjtZTPyHS">Discord</a>
</p>
