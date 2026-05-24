FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        git \
        build-essential \
        wget \
        libosmesa6-dev \
        libgl1 \
        libglew-dev \
        libglfw3 \
        patchelf \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /workspace

# ---- Stage 1: install PyTorch with CUDA 12.1 ----
# Pin torch to a version mbrl-lib is known to work with.
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# ---- Stage 2: SAC stack (stable-baselines3) ----
# SB3 2.3.x supports gymnasium 0.29 and torch 2.x
RUN pip install \
        stable-baselines3==2.3.2 \
        sb3-contrib==2.3.0 \
        gymnasium==0.29.1 \
        "gymnasium[mujoco]==0.29.1" \
        mujoco==3.1.6 \
        tensorboard

# ---- Stage 3: MBPO stack (mbrl-lib) ----
# mbrl-lib's pinned hydra-core==1.0.3 is too tight; install with --no-deps then fill in compat versions.
# pinning omegaconf 2.0.6 + hydra 1.1.2 works in practice for the mbpo example.
RUN pip install hydra-core==1.1.2 omegaconf==2.1.2 && \
    pip install --no-deps mbrl==0.2.0 && \
    pip install \
        tqdm \
        termcolor \
        colorlog \
        tensorboardX \
        scikit-learn \
        pytest \
        imageio

# ---- Stage 4: PILCO stack (TF 2.15 + GPflow 2.9) ----
# TF and PyTorch coexist fine; both pull their own CUDA runtime wheels.
RUN pip install \
        tensorflow[and-cuda]==2.15.1 \
        tensorflow-probability==0.23.0 \
        gpflow==2.9.2 \
        "numpy>=1.23.5,<2.0" \
        pandas \
        matplotlib \
        scipy

# ---- Project code ----
COPY . /workspace/

# Make PILCO importable
RUN pip install -e /workspace/pilco_src

# ---- Smoke import to fail the build fast on any broken pin ----
# Covers all the import paths the runners actually use, not just the top-level packages.
# Adding mbrl.algorithms.mbpo here is what catches missing transitive deps like tqdm.
RUN python -c "\
import torch, stable_baselines3, mbrl, tensorflow as tf, gpflow, gymnasium as gym, mujoco; \
import tensorflow_probability, omegaconf, hydra, tqdm; \
import mbrl.algorithms.mbpo, mbrl.util.common, mbrl.models; \
from mbrl.models import GaussianMLP; \
from pilco.models import PILCO; \
from pilco.controllers import RbfController; \
from pilco.rewards import ExponentialReward; \
print('all imports ok'); \
print('torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| mbrl', mbrl.__version__, '| tf', tf.__version__, '| gpflow', gpflow.__version__, '| gym', gym.__version__, '| mujoco', mujoco.__version__)\
"

CMD ["bash"]
