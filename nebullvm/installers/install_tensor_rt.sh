#!/bin/bash

if [[ "$(shell grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
  # Installation for centos type linux distribution
  # Try installation with pip if fails then install from source
  pip3 install --upgrade setuptools pip
  pip3 install nvidia-pyindex
  pip3 install --upgrade nvidia-tensorrt
  pip3 install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com

  if [[ $(python3 -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())" || echo 1) == 1 ]]
  then
    # Uninstall previous version
    pip3 uninstall nvidia-tensorrt
    # install pre-requisites
    pip3 install numpy
    yum update && \
      yum -y install glibnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev \
      libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer && \
      rm -rf /var/lib/apt/lists/*
  fi
else
  # Try installation with pip if fails then install from source
  pip install --upgrade setuptools pip
  pip install nvidia-pyindex
  pip install --upgrade nvidia-tensorrt
  pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com

  if [[ $(python3 -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())" || echo 1) == 1 ]]
  then
    # Uninstall previous version
    pip uninstall nvidia-tensorrt
    # install pre-requisites
    pip install numpy
    apt-get update && \
      apt-get -y install glibnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev \
      libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer && \
      rm -rf /var/lib/apt/lists/*
  fi
fi