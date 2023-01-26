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

By default, the `auto_installer` installs all the dl frameworks and compilers supported by `Speedster`, but some of these may not be relevant to your use case; therefore, in this section we explain how you can customise the installation of these libraries, avoiding those not needed.

For example, if you want to optimize a model with a PyTorch only optimization pipeline, you would not need to install TensorFlow and TensorFlow-specific compilers.

To address this need, `Speedster` has been designed to support many frameworks and optimisation techniques, but it is not mandatory to install them all. Each time it is used, `Speedster` will perform a check of the libraries installed in the environment, and will only use those that are available.

To customize the libraries installation you have two options:
- [Use the auto-installer (recommended)](#use-the-auto-installer-recommended)
- [Install the libraries manually](#manual-installation)

### Use the auto-installer (recommended)
To understand how to install only the required libraries, let's examine the auto-installer API:

    python -m nebullvm.installers.auto_installer --frameworks <frameworks> --extra-backends <backends> --compilers <compilers>

As you can see, three arguments are supported:

- `--frameworks` is used to specify the deep learning frameworks of the models that should be optimized by Speedster. The supported frameworks are: `torch`, `tensorflow`, `onnx` and `huggingface`. If you want to optimize a PyTorch model, here you should select `torch`. If included in this command, ONNX, TensorFlow and HuggingFace will be automatically installed by the auto-installer (if not already present in your env), while PyTorch must be installed manually before running this command. You can specify multiple frameworks by separating them with a space. For example, `--frameworks pytorch tensorflow` will set up your environment to optimize PyTorch and TensorFlow models. If you want to include all frameworks, you can use `all` as the argument. For example, `--frameworks all` will include all frameworks. Default: `all`.
- `--extra-backends` is used to specify the extra deep learning backends to be installed. Each framework specified in the previous command can exploit other dl frameworks (we name them backends) to boost the model performance. For example adding `onnx` as extra backend when optimizing a PyTorch model will enable the model conversion to ONNX and the optimization with all the ONNX supported compilers. The supported backends are: `torch`, `tensorflow` and `onnx`. The extra-backends that you select must be compatible with at least one of the frameworks selected with the previous option, please consult the table below to see the compatibility matrix. You can specify multiple backends by separating them with a space. For example, `--extra-backends tensorflow onnx` will install TensorFlow and ONNX. If you want to install all the backends supported by the selected frameworks, you can use `all` as the argument. For example, `--extra-backends all` will install all backends. If you don't want to install extra backends, you can set `--extra-backends none`. Default: `all`.
- `--compilers` is used to specify the deep learning compilers to be installed in addition to the frameworks. The supported compilers are: `deepsparse`, `tensor_rt`, `torch_tensor_rt`, `openvino` and `intel_neural_compressor`. The compilers that you select must be compatible with at least one of the frameworks/backends selected with the previous options, please consult the table below to see the compatibility matrix. You can specify multiple compilers by separating them with a space. For example, `--compilers deepsparse tensor_rt` will install DeepSparse and TensorRT. If you want to install all the compilers supported by the selected frameworks/backends, you can use `all` as the argument. For example, `--compilers all` will install all compilers. Speedster supports also `torchscript`, `tf_lite` and `onnxruntime`, but they are pre-installed in their respective frameworks (torchscript and tf_lite) or installed automatically if the framework is selected (onnxruntime), so you don't have to include them in this list. Speedster also supports `tvm`, which is not currently supported by the auto-installer and must be installed manually, see the next section if you want to include it.  Default: `all`.

Let's see an example of how to use these arguments:

    python -m nebullvm.installers.auto_installer --frameworks torch --extra-backends all --compilers all

This command will setup your environment to optimize PyTorch models, and will install all PyTorch supported backends and compilers.

The following table shows the supported combinations of frameworks, backends and compilers that you can install with the auto-installer:

| Framework   | Extra Backends            | Compilers                                                               |
|-------------|---------------------------|-------------------------------------------------------------------------|
| PyTorch     | ONNX                      | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |
| TensorFlow  | ONNX                      | TensorRT, OpenVINO                                                      |
| ONNX        | /                         | TensorRT, OpenVINO                                                      |
| HuggingFace | PyTorch, TensorFlow, ONNX | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |

!!! info
    HuggingFace models internally are based on either PyTorch or TensorFlow, depending on which model you choose. If that model is based on PyTorch, the framework must be included among the extra-backends, and the same applies to Tensorflow. ONNX, on the other hand, is optional and should be selected only if you want to include it in the optimization pipeline.

### Manual installation

If you want to manually install the requirements, this section collects links to the official installation guides for all frameworks and compilers supported by `Speedster`.

#### Deep Learning frameworks/backends
- PyTorch: https://pytorch.org/get-started/locally/
- TensorFlow: https://www.tensorflow.org/install
- ONNX: https://github.com/onnx/onnx#installation
- HuggingFace: https://huggingface.co/transformers/installation.html

#### Deep Learning compilers
- DeepSparse: https://github.com/neuralmagic/deepsparse#installation
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
- Torch TensorRT: https://pytorch.org/TensorRT/getting_started/installation.html#installation
- ONNXRuntime: https://onnxruntime.ai/docs/install/#python-installs
- OpenVINO: https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html#step-4-install-the-package
- Intel Neural Compressor: https://github.com/intel/neural-compressor#installation
- Apache TVM: https://tvm.apache.org/docs/install/index.html

#### Other requirements
- tf2onnx: https://github.com/onnx/tensorflow-onnx#installation (Install it if you want to convert TensorFlow models to ONNX)
- onnx-simplifier: https://github.com/daquexian/onnx-simplifier#python-version (Install it if you want to use TensorRT)
- onnxmltools: https://github.com/onnx/onnxmltools#install (Install it if you want to convert models to ONNX)

## (Optional) Download Docker images with frameworks and optimizers

Instead of installing the frameworks and compilers needed for optimization, which can be a time-consuming task, you can simply download a docker container with all compilers preinstalled.

To pull up the docker image, run:

    docker pull nebulydocker/nebullvm:latest

and then run and access the docker with:

    docker run -ti --gpus=all nebulydocker/nebullvm:latest

After optimizing the model, you may decide to deploy it to production. Note that you need to have the deep learning compiler used to optimize the model and other components inside the production docker. For this reason, we have created several versions of the Docker nebullvm container in the [Docker Hub](https://hub.docker.com/repository/docker/nebulydocker/nebullvm), each containing only one compiler. Pull the image with the compiler that has optimized your model!