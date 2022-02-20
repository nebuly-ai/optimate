#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [[ $OSTYPE == "darwin"* ]]
then
  brew install gcc git cmake
  brew install llvm
elif [[ "$(grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
  sudo yum update -y && sudo yum install -y gcc gcc-c++ llvm-devel cmake3 libxml2-dev
else
  apt-get update && apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
fi

if [ ! -d "tvm" ]
then
  git clone --recursive https://github.com/apache/tvm tvm
fi

cd tvm
mkdir -p build
cp $CONFIG_PATH build/
cd build
cmake ..
make -j8
cd ../python
python3 setup.py install --user
cd ../..
if [[ $OSTYPE == "darwin"* ]]
then
  brew install openblas gfortran
  pip install pybind11 cython pythran
  conda install -y scipy
  pip install xgboost
else
  pip3 install decorator attrs tornado psutil xgboost cloudpickle
fi