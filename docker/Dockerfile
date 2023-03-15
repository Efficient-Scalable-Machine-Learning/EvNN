FROM ubuntu:latest AS build
ENV MAX_JOBS=2 PIP_DEFAULT_TIMEOUT=100 DEBIAN_FRONTEND=noninteractive

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y software-properties-common build-essential gpg-agent
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y python3.8 python3.8-distutils python3-pip
# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Nvidia toolkit
RUN apt-get -y install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y cuda
#set up prerequisites for evnn compilation
RUN apt-get -y install python3-dev libpython3.8-dev libblas64-dev
RUN python -m pip install --upgrade pip
RUN python -m pip install ninja
RUN python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN apt-get clean && python -m pip cache purge