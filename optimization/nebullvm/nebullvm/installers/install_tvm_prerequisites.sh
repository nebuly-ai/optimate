#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [[ $OSTYPE == "darwin"* ]]
then
  brew install gcc git cmake
  #brew install llvm
  conda install -y -c conda-forge clangdev
elif [[ "$(grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
  sudo yum update -y && sudo yum install -y gcc gcc-c++ llvm-devel cmake3 git
  if [ -f "/usr/bin/cmake" ]
  then
    sudo alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake 10 \
      --slave /usr/local/bin/ctest ctest /usr/bin/ctest \
      --slave /usr/local/bin/cpack cpack /usr/bin/cpack \
      --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake \
      --family cmake
    sudo alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
      --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
      --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
      --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
      --family cmake
  else
    sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
  fi
else
  sudo apt-get update && sudo apt-get install -y libpython3.8 gcc libtinfo-dev zlib1g-dev \
    build-essential cmake libedit-dev libxml2-dev llvm-12
fi
