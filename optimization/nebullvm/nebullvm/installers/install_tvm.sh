#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

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
if [[ $OSTYPE == "darwin"* ]]
then
  pip install tornado
  brew install openblas gfortran
  pip install pybind11 cython pythran
  conda install -y scipy
  pip install xgboost decorator
  export MACOSX_DEPLOYMENT_TARGET=10.9
else
  pip3 install decorator attrs tornado psutil xgboost cloudpickle
fi
cd ../python
python3 setup.py install --user
cd ../..
