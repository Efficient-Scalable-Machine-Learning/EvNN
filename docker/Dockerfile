FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV EVNN_CUDA_COMPUTE 80
WORKDIR /evnn_src
COPY . .
RUN pip3 install .


WORKDIR /workspace