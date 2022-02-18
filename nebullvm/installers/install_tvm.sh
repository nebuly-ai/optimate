#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [[ "$(shell echo $$OSTYPE)" == "darwin"* ]]
then
  brew install gcc git cmake
  brew install llvm
elif [[ "$(shell grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
  yum update -y && yum install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
else
  apt-get update && apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
fi

if [ ! -d "tvm" ]
then
  git clone --recursive https://github.com/apache/tvm tvm
fi

git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir -p build
cp $CONFIG_PATH build/
cd build
cmake ..
make -j8
cd ../python
python setup.py install --user
cd ../..
pip install decorator attrs tornado psutil xgboost cloudpickle