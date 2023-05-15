FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --force-reinstall

COPY requirements.txt .

RUN python3.11 -m pip install -r requirements.txt

RUN python3.11 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
