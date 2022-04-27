#!/bin/bash

# Installation steps to build the ONNX-MLIR from source

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

# Build ONNX-MLIR

if [ ! -d "onnx-mlir" ]
then

  git clone --recursive https://github.com/onnx/onnx-mlir.git onnx-mlir
fi


if [ -z "$NPROC" ]
then
  NPROC=4
fi


# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

# Get the python interpreter path
export PYTHON_LOCATION=$(which python3)

mkdir onnx-mlir/build && cd onnx-mlir/build

if [[ -z "$PYTHON_LOCATION" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  echo "Using python path " $PYTHON_LOCATION
  echo "Using MLIR_DIR " $MLIR_DIR

  cmake -G Ninja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=${PYTHON_LOCATION} \
      -DPython3_EXECUTABLE=${PYTHON_LOCATION} \
      -DMLIR_DIR=${MLIR_DIR} \
      ..

fi

cmake --build . --parallel $NPROC

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --parallel $NPROC --target check-onnx-lit
