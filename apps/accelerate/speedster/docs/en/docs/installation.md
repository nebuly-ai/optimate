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

To customize the libraries installation you have two options:
- [Use the auto-installer (recommended)](#use-the-auto-installer-recommended)
- [Install the libraries manually](#manual-installation)

### Use the auto-installer (recommended)
To understand how to install only the required libraries, let's examine the auto-installer API:

    python -m nebullvm.installers.auto_installer --frameworks <frameworks> --backends <backends> --compilers <compilers>

As you can see, three arguments are supported:

- `--frameworks` is used to specify the deep learning frameworks to be installed. The supported frameworks are: `torch`, `tensorflow`, `onnx` and `huggingface`. If you want to optimize a PyTorch model, here you should select `torch`. You can specify multiple frameworks by separating them with a space. For example, `--frameworks pytorch tensorflow` will install PyTorch and TensorFlow. If you want to install all frameworks, you can use `all` as the argument. For example, `--frameworks all` will install all frameworks. Default: `all`.
- `--backends` is used to specify the deep learning backends to be installed. Each framework specified in the previous command can exploit other dl frameworks (we name them backends) to boost the model performance, for example adding `onnx` as backend when optimizing a PyTorch model will enable the model conversion to ONNX and the optimization with all the ONNX supported compilers. The supported backends are: `torch`, `tensorflow` and `onnx`. You can specify multiple backends by separating them with a space. For example, `--backends torch tensorflow` will install PyTorch and TensorFlow. If you want to install all backends, you can use `all` as the argument. For example, `--backends all` will install all backends. Default: `all`.
- `--compilers` is used to specify the deep learning compilers to be installed in addition to the frameworks. The supported compilers are: `deepsparse`, `tensor_rt`, `torch_tensor_rt`, `onnxruntime`, `openvino`and `intel_neural_compressor`. You can specify multiple compilers by separating them with a space. For example, `--compilers deepsparse tensor_rt` will install DeepSparse and TensorRT. If you want to install all compilers, you can use `all` as the argument. For example, `--compilers all` will install all compilers. Speedster supports also `torchscript` and `tf_lite` compilers, but they are pre-installed inside the frameworks, so you don't have to include them in this list. Speedster also supports `tvm`, which is not currently supported by the auto-installer and must be installed manually, see the next section if you want to include it.  Default: `all`.

Let's see an example of how to use these arguments:

    python -m nebullvm.installers.auto_installer --frameworks torch --backends all --compilers all

This command will install PyTorch, as well as all PyTorch supported backends and compilers.

The following table shows the supported combinations of frameworks, backends and compilers that you can install with the auto-installer:

| Framework   | Backends                  | Compilers                                                                                        |
|-------------|---------------------------|--------------------------------------------------------------------------------------------------|
| PyTorch       | ONNX                      | DeepSparse, TensorRT, Torch TensorRT, ONNXRuntime, OpenVINO, Intel Neural Compressor |
| TensorFlow  | ONNX                      | ONNXRuntime, TensorRT, OpenVINO                                                      |
| ONNX        | /                         | ONNXRuntime, TensorRT, OpenVINO                                                     |
| HuggingFace | PyTorch, TensorFlow, ONNX | DeepSparse, TensorRT, Torch TensorRT, ONNXRuntime, OpenVINO, Intel Neural Compressor           |

### Manual installation

If you want to install the requirements manually, this section collects the links to the official installation guides of all the frameworks and compilers supported by `Speedster`.

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


## (Optional) Download Docker images with frameworks and optimizers

Instead of installing the frameworks and compilers needed for optimization, which can be a time-consuming task, you can simply download a docker container with all compilers preinstalled.

To pull up the docker image, run:

    docker pull nebulydocker/nebullvm:latest

and then run and access the docker with:

    docker run -ti --gpus=all nebulydocker/nebullvm:latest

After optimizing the model, you may decide to deploy it to production. Note that you need to have the deep learning compiler used to optimize the model and other components inside the production docker. For this reason, we have created several versions of the Docker nebullvm container in the [Docker Hub](https://hub.docker.com/repository/docker/nebulydocker/nebullvm), each containing only one compiler. Pull the image with the compiler that has optimized your model!