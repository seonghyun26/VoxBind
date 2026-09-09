# syntax=docker/dockerfile:1.6
#
# VoxBind ENVIRONMENT image — reproduces the locked `voxbind` conda env exactly
# (env/voxbind.conda-linux-64.lock + env/voxbind.pip.lock.txt). This single env
# runs the GPU train/sample pipeline AND Vina docking + chem/geometry eval.
# vina is 1.2.7 (this env is py3.10; vina 1.2.2 has no py3.10 wheel). For the
# paper-faithful vina 1.2.2 build a separate py3.8 env — see script/00_setup_env.sh.
# (PoseCheck/PoseBusters pose eval lives in a separate `moleval` env — not here.)
#
# torch is a pip wheel that bundles its own CUDA runtime, so the base image needs
# NO CUDA — the host just needs an NVIDIA driver (>= the torch wheel's CUDA, ~550
# for cu12.x) + the NVIDIA Container Toolkit. Run with `--gpus all`.
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
#
# NOTE: the conda lock is linux-64. On another arch, fall back to `env.yaml`.

ARG MICROMAMBA_VERSION=1.5.10
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# git → for the AutoDockTools_py3 VCS pip pin; X libs → rdkit/openbabel.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git ca-certificates libxrender1 libxext6 libsm6 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /workspace/VoxBind

# 1) conda layer — byte-exact from the explicit lock. No solver, just fetch+link.
#    Copy only the lock first so this heavy layer caches across code changes.
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.conda-linux-64.lock /tmp/voxbind.conda-linux-64.lock
RUN micromamba create -y -n voxbind --file /tmp/voxbind.conda-linux-64.lock \
 && micromamba clean --all --yes

# 2) pip layer — torch, vina 1.2.2, meeko, pdb2pqr, AutoDockTools, pyuul, hydra, …
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.pip.lock.txt /tmp/voxbind.pip.lock.txt
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba run -n voxbind pip install --no-cache-dir -r /tmp/voxbind.pip.lock.txt

# 3) project source (data/weights excluded via .dockerignore) + editable install.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/VoxBind
RUN micromamba run -n voxbind pip install --no-cache-dir -e .

# Activate voxbind for every interactive shell / RUN.
ENV ENV_NAME=voxbind \
    MAMBA_DEFAULT_ENV=voxbind \
    HOME=/home/$MAMBA_USER \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

CMD ["bash"]
