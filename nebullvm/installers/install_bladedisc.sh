#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [ ! -d "BladeDISC" ]
then
  git clone https://github.com/alibaba/BladeDISC.git
fi

cd BladeDISC && git submodule update --init --recursive

if [ $1 == "true" ]
then
cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh
else
  if [[ $OSTYPE == "darwin"* ]]
  then
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.10.0+aarch64
    cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh
  else
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.8.1+cpu
    cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh
  fi
fi

cd ../..
