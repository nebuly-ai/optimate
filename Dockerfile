ARG STARTING_IMAGE=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM ${STARTING_IMAGE}

# Set frontend as non-interactive
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade

# Install python and pip
RUN apt-get install -y python3-opencv python3-pip && \
    python3 -m pip install --upgrade pip && \
    apt-get -y install git && \
    apt-get -y install python-is-python3

# Install other libraries
RUN apt-get install -y sudo wget

# Install dl frameworks
RUN pip3 install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install --no-cache-dir tensorflow
RUN pip3 install --no-cache-dir onnx
RUN pip3 install --no-cache-dir transformers

# Copy the working dir to the container
COPY . /nebullvm

# Install nebullvm
ARG NEBULLVM_VERSION=latest
RUN if [ "$NEBULLVM_VERSION" = "latest" ] ; then \
        cd nebullvm ; \
        pip install . ; \
        cd apps/accelerate/speedster ; \
        pip install . ; \
        cd ../../../.. ; \
        rm -rf nebullvm ; \
    else \
        pip install --no-cache-dir nebullvm==${NEBULLVM_VERSION} ; \
    fi

# Install required python modules
RUN pip install --no-cache-dir cmake

# Install default deep learning compilers
ARG COMPILER=all
RUN if [ "$COMPILER" = "all" ] ; then \
        python3 -m nebullvm.installers.auto_installer --frameworks all --extra-backends all --compilers all ; \
    elif [ "$COMPILER" = "tensorrt" ] ; then \
        python3 -m nebullvm.installers.auto_installer --frameworks all --extra-backends all --compilers tensorrt ; \
    elif [ "$COMPILER" = "openvino" ] ; then \
        python3 -m nebullvm.installers.auto_installer --frameworks all --extra-backends all --compilers openvino ; \
    elif [ "$COMPILER" = "onnxruntime" ] ; then \
        python3 -m nebullvm.installers.auto_installer --frameworks all --extra-backends all --compilers onnxruntime ; \
    fi

# Install TVM
RUN if [ "$COMPILER" = "all" ] || [ "$COMPILER" = "tvm" ] ; then \
        pip install --no-cache-dir https://github.com/tlc-pack/tlcpack/releases/download/v0.10.0/apache_tvm_cu116_cu116-0.10.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ; \
        pip install --no-cache-dir xgboost ; \
        python3 -c "from tvm.runtime import Module" ; \
    fi

ENV SIGOPT_PROJECT="tmp"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/tensorrt
