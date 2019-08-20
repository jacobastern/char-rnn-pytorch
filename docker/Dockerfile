FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# install build tools & python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    build-essential \
    cmake \
    curl \
    g++ \
    git \
    graphicsmagick \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libfreetype6-dev \
    libgraphicsmagick1-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libpng-dev \
    libswscale-dev \
    libzmq3-dev \
    pkg-config \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-lxml \
    python3-setuptools \
    rsync \
    software-properties-common \
    unzip \
    wget \
    vim \
    zip

# install cleanup
RUN apt-get -qq update && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/*

# update python installers
RUN pip3 install pip --upgrade

# install python dependencies
RUN pip3 --no-cache-dir install \
    argparse \
    opencv-python \
    sklearn \
    cython \
    ipykernel \
    jupyter \
    matplotlib \
    numpy \
    seaborn \
    pprint \
    requests \
    scipy \
    setuptools \    
    torchvision \
    xgboost \
    h5py \
    torchtext \ 
    torch==1.1.0 -f https://download.pytorch.org/whl/cu100/stable \
    && python3 -m ipykernel.kernelspec

#RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENTRYPOINT ["/bin/bash"]

WORKDIR /code
