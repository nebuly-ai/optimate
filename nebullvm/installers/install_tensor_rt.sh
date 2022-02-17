#!/bin/bash

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