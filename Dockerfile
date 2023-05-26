FROM syrkis/hpc

COPY requirements.txt .

RUN python -m pip install --no-cache-dir  -r requirements.txt

RUN python -m pip install --upgrade \
    "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

