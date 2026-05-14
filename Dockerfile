# syntax=docker/dockerfile:1.6
#
# VoxBind container.
# Mirrors README: micromamba env from env.yaml, then `pip install -e .`.
# CUDA runtime libs come from pytorch-cuda inside the env, so the host only
# needs an NVIDIA driver new enough for CUDA 11.8 (>= 520) and the NVIDIA
# Container Toolkit. Run with `--gpus all`.
#
# Build:    docker build -t voxbind .
# Slim:     docker build --build-arg ENV_FILE=env.minimal.yaml -t voxbind:min .
# Run:      docker run --rm -it --gpus all --shm-size=16g \
#               -v $PWD/dataset/data:/workspace/dataset/data \
#               -v $PWD/voxbind/exps:/workspace/voxbind/exps \
#               -v $PWD/voxbind/log:/workspace/voxbind/log \
#               voxbind

ARG MICROMAMBA_VERSION=1.5.10
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# System libs needed by openbabel/rdkit GUI deps and git for any vcs pip pkgs.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libxrender1 \
        libxext6 \
        libsm6 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /workspace

# Solve the env first so this layer caches across code changes.
ARG ENV_FILE=env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER ${ENV_FILE} /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml \
 && micromamba clean --all --yes

# Activate the base env for subsequent RUN/CMD/ENTRYPOINT.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Project source + editable install.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace
RUN pip install --no-cache-dir -e .

ENV HOME=/home/$MAMBA_USER \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

CMD ["bash"]
