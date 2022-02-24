
<img src="https://user-images.githubusercontent.com/83510798/155317935-523dcf79-9adb-4131-9511-8e269a1f97dd.png">

# Nebullvm

> _**All-in-one library that allows you to test multiple DL compilers in one line of code and speed up the inference of your DL models by 5-20 times.**_


This repository contains the opensource `nebullvm` package, an opensource project
aiming to reunite all the opensource AI compilers under the same easy-to-use interface.

You will love this library if:<br />
🚀 you want to speed up the response time of your AI models;\
👟 you don't want to test all DL compilers on the market, 
but you just want to know the best one for your specific application;\
🥇 you enjoy simplifying complex problems: in fact with one 
line of code you can know which DL compiler is best suited for your 
application;\
💙 you are passionate about AI performance optimization.

We designed something that is super easy to use: you just need to 
input your DL model and you will automatically get back an 
optimized version of the model for the hardware where you performed 
the optimization.

_To the best of the authors' knowledge, there are no open source 
libraries yet to combine the various DL compilers on the market to 
figure out which one is best suited for the user's model. 
We believe that this library can make a strong contribution to making it 
increasingly easy for AI developers to make their models more efficient 
without spending an inordinate amount of time on it._

## Supported frameworks
- PyTorch
- TensorFlow
- Hugging Face (Coming soon 🤗)

## Supported DL compilers
- OpenVINO
- TensorRT
- TVM
- MLIR (Coming soon 🚀)

## Installation
There are two ways to install `nebullvm`: 
1. Using PyPi. We suggest installing the library with `pip` to get the 
stable version of `nebullvm`
2. From source code to get the latest features


### Source code installation
To install the source code you have to clone the directory on your local 
machine using `git`.
```
git clone https://github.com/nebuly-ai/nebullvm.git
```
Then, enter the repo and install `nebullvm` with `pip`.
```
cd nebullvm
pip install .
```
### Installation with PyPi
The easiest way to install `nebullvm` is by using `pip`, running
```
pip install nebullvm
```

### Auto-Installer
Auto-installer is a new feature for automatic installation of all the DL compilers
supported by `nebullvm`. For activating it, it is enough to import `nebullvm` and 
our code will automatically take care of the installation. We highly recommend
running
```
python -c "import nebullvm"
```
before starting using the library for the first time. In this way you'll avoid any import issue
due to libraries needing the reboot of the python kernel before being used. 
You should ignore at this stage any import error 
resulting on the previous command.

The library automatically installs the available DL compilers. However, 
if you prefer to avoid automatic installation, you can simply export the 
environment variable `NO_COMPILER_INSTALLATION=1` by running
```
export NO_COMPILER_INSTALLATION=1
```
from your command line or adding
```
import os
os.environ["NO_COMPILER_INSTALLATION"] = "1"
```
in your python code before importing `nebullvm` for the first time.

Note that auto-installation of opensource compilers is done outside 
the nebullvm wheel. Installations of ApacheTVM and Openvino have been 
tested on macOS, linux distributions similar to Debian and CentOS. 

The feature is still in an alpha version, so we expect that it may fail 
under untested circumstances. We are doing our best to support as many 
hardware configurations, operating systems, and development frameworks 
as possible, so we strongly encourage you to open a new github-issue 
whenever you encounter a bug or failure of the library.

### Possible issues
**MacOS**: the installation may fail on MacOS for MacBooks with the Apple Silicon chip,
due to scipy compilation errors. The easy fix is to install scipy with another package 
manager such as conda (the Apple Silicon distribution of Mini-conda) 
and then install `nebullvm`.
For any additional issues do not hesitate to open an issue or contact directly 
`info@nebuly.ai` by email.

## Get Started
`Nebullvm` aims to reduce the computation time of deep learning model 
inference by at least 5 times by selecting and using the right DL compiler 
for your specific model and machine.
Currently `nebullvm` supports models in the `torch` and `tensorflow` frameworks, 
but others will be included soon. 
Templates can be easily imported from one of the supported frameworks using the 
appropriate function.

Here we present an example of optimizing a pytorch model with `nebullvm`:
```
>>> import torch
>>> import torchvision.models as models
>>> from nebullvm import optimize_torch_model
>>> model = models.efficientnet_b0()
>>> bs, input_sizes = 1, [(3, 256, 256)]
>>> save_dir = "."
>>> optimized_model = optimize_torch_model(
...     model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir
... )
>>> x = torch.randn((bs, *input_sizes[0]))
>>> res = optimized_model(x)
```
The same optimization can be achieved with a `tensorflow` model using the function 
`nebullvm.optimize_tf_model`.


## Acknowledgments

This library is based on the amazing work done on AI compilers by 
the opensource community and major hardware suppliers. 
The purpose of `nebullvm` is to combine the incredible work made so far 
into a common, easy-to-use interface for ML developers.

Currently `nebullvm` supports as AI compilers:
* OpenVINO (on Intel Machines)
* TensorRT (on Nvidia GPUs)
* Apache TVM
