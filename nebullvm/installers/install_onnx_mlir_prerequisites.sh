#!/bin/bash

# Installation steps to build and install the llvm-project from source

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [ -z "$NPROC" ]
then
  export NPROC=4
fi

# Install the OS dependent required packages
if [[ $OSTYPE == "darwin"* ]]
then
  brew install gcc git cmake ninja pybind11
elif [[ "$(grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
    sudo yum update -q -y && \
    sudo yum install -q -y \
        autoconf automake ca-certificates cmake diffutils \
        file java-11-openjdk-devel java-11-openjdk-headless \
        gcc gcc-c++ git libtool make ncurses-devel \
        zlib-devel && \
    # Install ninja
    git clone -b v1.10.2 https://github.com/ninja-build/ninja.git && \
    cd ninja && mkdir -p build && cd build && \
    cmake .. && \
    make -j$NPROC install && \
    cd ../.. && rm -rf ninja;
else
  sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    autoconf automake ca-certificates cmake curl \
    default-jdk-headless gcc g++ git libncurses-dev \
    libtool make maven ninja-build openjdk-11-jdk-headless \
    zlib1g-dev

fi

# Install protobuf
PROTOBUF_VERSION=3.14.0
git clone -b v$PROTOBUF_VERSION --recursive https://github.com/google/protobuf.git \
    && cd protobuf && ./autogen.sh \
    && ./configure --enable-static=no \
    && make -j$NPROC install && ldconfig \
    && cd python && python setup.py install \
    && cd ../.. && rm -rf protobuf

# Install jsoniter
JSONITER_VERSION=0.9.23
JSONITER_URL=https://repo1.maven.org/maven2/com/jsoniter/jsoniter/$JSONITER_VERSION \
    && JSONITER_FILE=jsoniter-$JSONITER_VERSION.jar \
    && curl -s $JSONITER_URL/$JSONITER_FILE -o /usr/share/java/$JSONITER_FILE


# ONNX-MLIR needs the llvm-project build from the source

# Firstly, install MLIR (as a part of LLVM-Project):
git clone -n https://github.com/llvm/llvm-project.git


# Check out a specific branch that is known to work with ONNX-MLIR.
# TBD: Option to set the commit hash dynamically
cd llvm-project && git checkout a7ac120a9ad784998a5527fc0a71b2d0fd55eccb && cd ..

mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . --parallel $NPROC -- ${MAKEFLAGS}
cmake --build . --parallel $NPROC --target check-mlir
