# syntax=docker/dockerfile:1.6
#
# VoxBind ENVIRONMENT image — restores the conda-lock YAML locks exactly.
# Two envs, both from *.conda-lock.yml (conda + pip, hash-pinned, linux-64):
#   voxbind  (py3.10)  GPU train/sample pipeline + chem/geometry eval + vina 1.2.7
#   voxdock  (py3.8)   AutoDock Vina 1.2.2 docking (the paper's numbers)
# 05_evaluate.sh runs metrics.py under voxdock (vina 1.2.2). PoseCheck/PoseBusters
# pose eval lives in a separate `moleval` env — build with 00_setup_env.sh if needed.
#
# pytorch-cuda 11.8 comes from the lock, so the host only needs an NVIDIA driver
# (>= 520 for CUDA 11.8) + the NVIDIA Container Toolkit. Run with `--gpus all`.
#
# BUILD FROM THE REPO ROOT (context = repo root; .dockerignore keeps data/weights out):
#   docker build -f env/Dockerfile -t voxbind:env .
#
# RUN (mount data/weights at runtime — they are NOT in the image):
#   docker run --rm -it --gpus all --shm-size=16g \
#     -v $PWD/voxbind/dataset/data:/workspace/VoxBind/voxbind/dataset/data \
#     -v $PWD/voxbind/model_zoo:/workspace/VoxBind/voxbind/model_zoo \
#     -v $PWD/voxbind/exps:/workspace/VoxBind/voxbind/exps \
#     voxbind:env

ARG MICROMAMBA_VERSION=1.5.10
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# git → AutoDockTools_py3 VCS pip pin; X libs → rdkit/openbabel.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates libxrender1 libxext6 libsm6 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /workspace/VoxBind

# conda-lock as the installer (uses micromamba as the solver).
RUN micromamba install -y -n base -c conda-forge conda-lock \
 && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy just the locks first so these heavy layers cache across code changes.
# `conda-lock install` restores conda AND pip deps from each YAML lock.
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.conda-lock.yml /tmp/voxbind.conda-lock.yml
RUN micromamba run -n base conda-lock install --micromamba -n voxbind /tmp/voxbind.conda-lock.yml \
 && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxdock.conda-lock.yml /tmp/voxdock.conda-lock.yml
RUN micromamba run -n base conda-lock install --micromamba -n voxdock /tmp/voxdock.conda-lock.yml \
 && micromamba clean --all --yes

# TargetDiff sibling (docking eval imports its utils.evaluation.* + full receptors).
RUN git clone --depth 1 https://github.com/guanjq/targetdiff /workspace/targetdiff

# Project source (data/weights excluded via .dockerignore) + editable install.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/VoxBind
RUN micromamba run -n voxbind pip install --no-cache-dir -e .

ENV ENV_NAME=voxbind \
    MAMBA_DEFAULT_ENV=voxbind \
    HOME=/home/$MAMBA_USER \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    TARGETDIFF_ROOT=/workspace/targetdiff \
    FULL_RECEPTOR_ROOT=/workspace/targetdiff/data/test_set/test_set

CMD ["bash"]
