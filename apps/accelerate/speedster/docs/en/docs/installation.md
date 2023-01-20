# Installation
In this installation guide we will learn:

- [Quick installation](#quick-installation) of `Speedster` with pip **(Recommended)** 

- [Selective installation](#optional-selective-installation-of-speedster-requirements) of the requirements **(Optional)** 

- [Installation](#optional-download-docker-images-with-frameworks-and-optimizers) with docker **(Optional)** 

## Quick installation 
You can easily install `Speedster` using pip.

    pip install speedster

Then make sure to install all the available deep learning compilers:

    python -m nebullvm.installers.auto_installer --compilers all


!!! info
    If you want to optimize PyTorch or HuggingFace models, PyTorch must be pre-installed in the environment before using the auto-installer, please install it from [this](https://pytorch.org/get-started/locally/) link. Moreover, for Mac computers with M1/M2 processors, please use a conda environment, or you may run into problems when installing some of the deep learning compilers.

Great, now you are now ready to accelerate your model 🚀 Please visit the following pages to get started based on the DL framework of your input model:

- [Getting started with PyTorch optimization](./getting_started/pytorch_getting_started.md)
- [Getting started with 🤗 HuggingFace optimization](./getting_started/hf_getting_started.md)
- [Getting started with TensorFlow/Keras optimization](./getting_started/tf_getting_started.md)
- [Getting started with ONNX optimization](./getting_started/onnx_getting_started.md)


## (Optional) Selective installation of Speedster requirements

TODO

## (Optional) Download Docker images with frameworks and optimizers

Instead of installing the frameworks and compilers needed for optimization, which can be a time-consuming task, you can simply download a docker container with all compilers preinstalled.

To pull up the docker image, run:

    docker pull nebulydocker/nebullvm:latest

and then run and access the docker with:

    docker run -ti --gpus=all nebulydocker/nebullvm:latest

After optimizing the model, you may decide to deploy it to production. Note that you need to have the deep learning compiler used to optimize the model and other components inside the production docker. For this reason, we have created several versions of the Docker nebullvm container in the [Docker Hub](https://hub.docker.com/repository/docker/nebulydocker/nebullvm), each containing only one compiler. Pull the image with the compiler that has optimized your model!