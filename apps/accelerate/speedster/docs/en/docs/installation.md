# Installation
In this installation guide you will learn:

- Quick installation of Speedster with pip

- **(Optional)** Selective installation of the requirements

- **(Optional)** Installation with docker

## Quick installation 
You can easily and rapidly install Speedster using pip.

    pip install speedster

Then make sure to install the deep learning compilers to leverage during the optimization:

    python -m nebullvm.installers.auto_installer --backends all --compilers all

Alternatively, you can install Speedster from source code to take advantage of the latest features.

    pip install git+https://github.com/nebuly-ai/nebullvm.git#subdirectory=apps/accelerate/speedster

!!! info
    For Mac computers with M1/M2 processors, please use a conda environment, or you may run into problems when installing some of the deep learning compilers.
    Moreover, if you want to optimize PyTorch or HuggingFace models, PyTorch must be pre-installed in the environment before using the auto-installer, please install it from this link.


Great, you are now ready to accelerate your model 🚀 Let's move on to the Getting started.
