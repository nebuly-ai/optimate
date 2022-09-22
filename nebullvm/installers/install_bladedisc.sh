#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [ ! -d "BladeDISC" ]
then
  git clone https://github.com/alibaba/BladeDISC.git
fi

cd BladeDISC && git submodule update --init --recursive

# Install bazel
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
sudo apt install default-jdk

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
