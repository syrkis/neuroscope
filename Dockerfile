FROM syrkis/hpc

COPY requirements.txt .

RUN python -m pip install --no-cache-dir  -r requirements.txt

RUN python -m pip install --upgrade \
    "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu