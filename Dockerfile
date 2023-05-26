FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y \
    python3.11 python3.11-distutils python3.11-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --force-reinstall && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3.11 -m pip install --no-cache-dir  -r requirements.txt

RUN python3.11 -m pip install --upgrade \
        "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python3.11 -m pip install torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
