# Installation
In this installation guide you will learn:

- [Quick installation](## Quick installation ) of `Speedster` with pip **(Recommended)** 

- [Selective installation](## (Optional) Selective installation of Speedster requirements) of the requirements **(Optional)** 

- [Installation](## Download Docker images with frameworks and optimizers) with docker **(Optional)** 

## Quick installation 
You can easily install `Speedster` using pip.

    pip install speedster

Then make sure to install all the available deep learning compilers:

    python -m nebullvm.installers.auto_installer --compilers all


!!! info
    For Mac computers with M1/M2 processors, please use a conda environment, or you may run into problems when installing some of the deep learning compilers.
    Moreover, if you want to optimize PyTorch or HuggingFace models, PyTorch must be pre-installed in the environment before using the auto-installer, please install it from this link.

Great, you are now ready to accelerate your model 🚀 Let's move on to the Getting started.

## (Optional) Selective installation of Speedster requirements
As an alternative of the Quick installation of Speedster, in this section we explain how to selectively install the requirements and avoid the installation of libraries that are not needed for your use case. 

!!! info
    We refer to PyTorch, ONNX, TensorFlow or HuggingFace as optimization backends and to TensorRT, ONNX Runtime, Openvino, etc. as optimizers.
    
In principle not all the backends and compilers that would be installed may be relevant for a specific use case. For example, if you want to optimize a model with a PyTorch-only optimization pipeline, you would not need to install TensorFlow and TensorFlow-specific backends and optimizers.

To selectively install Speedster requirements, there are 2 options:

- [Use the auto-installer](### Use the auto-installer) (recommended)
- Manual installation

### Use the auto-installer (recommended)

Speedster's auto_installer can be used with the command:

    python -m nebullvm.installers.auto_installer --backends fr1 fr2 --compilers comp1 comp2

The supported arguments are the following:

- `backends`: list of deep learning backends you want to support among torch, onnx, tensorflow and huggingface. By default, this argument is set to "all". Each framework includes a base option and a full option. With the base option, only the single selected framework will be installed, while with the full option also all the other frameworks that support conversion starting from the selected framework will be included. The following table shows the full list of supported options:

TODO

## Download Docker images with frameworks and optimizers