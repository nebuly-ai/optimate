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

By default, the `auto_installer` installs all the DL frameworks and compilers supported by `Speedster`. However, some of these may not be relevant to your use case. In this section, we explain how you can customize the installation of these libraries, avoiding those that are not needed.

To customize the libraries installation you have two options:

- [Use the auto-installer (recommended)](#use-the-auto-installer-recommended)
- [Install the libraries manually](#manual-installation)

### Use the auto-installer (recommended)
To understand how to selectively install your preferred libraries, let's examine the auto-installer API:

```python
python -m nebullvm.installers.auto_installer 
    --frameworks <frameworks> 
    --extra-backends <backends> 
    --compilers <compilers>
```

!!! Description

    === "--frameworks"

        `frameworks` is used to specify the deep learning framework of your input model. The supported frameworks are `torch`, `tensorflow`, `onnx` and `huggingface`.

        - if you want to optimize a model with a single DL framework, the code is as follows (example below for HuggingFace):
            
            ```python
            python -m nebullvm.installers.auto_installer --frameworks huggingface
            ```
            
            Please remember that for PyTorch optimization, you should pre-install PyTorch from the official [repo](https://pytorch.org/get-started/locally/).
                
        - if you want to optimize models in multiple input frameworks, you can include them separated with a space:
            ```python
            python -m nebullvm.installers.auto_installer --frameworks tensorflow torch
            ```

        - If you want to include all the frameworks, you can use `all` as the argument:

            ```python
            python -m nebullvm.installers.auto_installer --frameworks all
            ```

        Default: `all`.
    
    === "--extra-backends"

        After entering your input model, `Speedster` converts the input model from its original framework into an intermediate framework to be used during the optimization; we call these intermediate frameworks "backends." To learn more, see the section [Model Converter](https://docs.nebuly.com/Speedster/key_concepts/) in the docs. This conversion allows `Speedster` to apply all optimization techniques without being constrained by the input framework of your model.
            
        The supported backends are `torch`, `tensorflow` and `onnx`.
            
        You can specify multiple backends by separating them with a space. 
            
        - For example, if you want to install TensorFlow and ONNX as backends of an HugginFace model, the code is as follows:
            
            ```python
            python -m nebullvm.installers.auto_installer --frameworks huggingface --extra-backends tensorflow onnx
            ```python
            
        - If you want to install all the backends supported by the selected frameworks, you can use `all` as the argument.
        - If you don't want to install extra backends, you can set `--extra-backends none`.
            
        The extra-backends that you choose must be compatible with at least one of the input frameworks you previously selected with the argument `—-frameworks`, please see the table below to see the compatibility matrix. 

        Default: `all`.    

    === "--compilers"

        `compilers` is used to specify the deep learning compilers to be installed. The supported compilers are: `deepsparse`, `tensor_rt`, `torch_tensor_rt`, `openvino` and `intel_neural_compressor`. The compilers must be compatible with at least one of the backends selected with the argument `—-extra-backends`, please see the table below to see the compatibility matrix.

        - You can specify multiple compilers by separating them with a space. For example:
            
            ```python
            --compilers deepsparse tensor_rt
            ```
            
            will install DeepSparse and TensorRT. 
            
        - If you want to install all the compilers supported by the selected frameworks/backends, you can use `all` as the argument.

        Speedster also supports `torchscript`, `tf_lite`, and `onnxruntime` as built-in; these are preinstalled with their respective backends, so there is no need to include them in the list. Speedster also supports `tvm`, which is currently not supported by the automatic installer and must be installed manually; see the next section if you wish to include it.

        Default: `all`.


Let's see an example of how to use these three arguments:

```python
python -m nebullvm.installers.auto_installer 
    --frameworks torch 
    --extra-backends all 
    --compilers all
```

This command will setup your environment to optimize PyTorch models, and will install all PyTorch supported backends and compilers.

The following table shows the supported combinations of frameworks, backends and compilers that you can install with the auto-installer:

| Framework   | Extra Backends            | Compilers                                                               |
|-------------|---------------------------|-------------------------------------------------------------------------|
| PyTorch     | ONNX                      | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |
| TensorFlow  | ONNX                      | TensorRT, OpenVINO                                                      |
| ONNX        | /                         | TensorRT, OpenVINO                                                      |
| HuggingFace | PyTorch, TensorFlow, ONNX | DeepSparse, TensorRT, Torch TensorRT, OpenVINO, Intel Neural Compressor |

!!! info
    Hugginface models can be of two types, PyTorch-based or TensorFlow-based. For PyTorch-based models, it is necessary to include `torch` as an extra-backend. For TensorFlow-based models, you must include `tensorflow` as an extra-backend.

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