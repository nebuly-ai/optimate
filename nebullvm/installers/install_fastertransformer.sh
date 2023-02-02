#!/bin/bash

# TODO: check requirements
# https://github.com/NVIDIA/FasterTransformer/blob/main/docs/bert_guide.md
# Requirements
#CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
#CUDA 11.0 or newer version
#Python: Only verify on python 3
#Tensorflow: Verify on 1.15, 1.13 and 1.14 should work.
#PyTorch: Verify on 1.8.0, >= 1.5.0 should work.


# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [[ $OSTYPE == "darwin"* ]]
then
  echo "MacOS is not supported for FasterTransformer"
  exit 1
fi

if [ ! -d "FasterTransformer" ]
then
  git clone --recursive https://github.com/NVIDIA/FasterTransformer FasterTransformer
fi

# TODO: checkout to latest release

cd FasterTransformer &&
mkdir -p build &&
cd build &&
cmake -DSM=$COMPUTE_CAPABILITY -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=OFF .. &&
make -j8

# TODO: enable multi gpu if possible