ARG STARTING_IMAGE=nvcr.io/nvidia/tensorrt:23.03-py3
FROM ${STARTING_IMAGE}

WORKDIR /

# Set frontend as non-interactive
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install other libraries
RUN apt-get install -y sudo wget

# Install libraries
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118  \
    && pip install --no-cache-dir tensorflow \
    && pip install --no-cache-dir xformers \
    && pip install --no-cache-dir accelerate \
    && python3 -m pip install --no-cache-dir --upgrade tensorrt

# Copy the working dir to the container
COPY ../.. /nebullvm

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
        pip install --no-cache-dir https://github.com/tlc-pack/tlcpack/releases/download/v0.11.1/tlcpack_cu116-0.11.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ; \
        pip install --no-cache-dir xgboost ; \
        python3 -c "from tvm.runtime import Module" ; \
    fi

ENV SIGOPT_PROJECT="tmp"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/tensorrt
ENV CUDA_MODULE_LOADING="LAZY"
