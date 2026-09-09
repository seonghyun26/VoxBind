# syntax=docker/dockerfile:1.6
#
# VoxBind ENVIRONMENT image — reproduces the locked conda envs exactly.
# Two envs, both from lock files, so docking is paper-faithful:
#   voxbind  (py3.10)  GPU train/sample pipeline + chem/geometry eval + vina 1.2.7
#   voxdock  (py3.8)   AutoDock Vina 1.2.2 docking (the paper's numbers)
# 05_evaluate.sh runs metrics.py under voxdock (vina 1.2.2). PoseCheck/PoseBusters
# pose eval lives in a separate `moleval` env — build it with 00_setup_env.sh if needed.
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
# NOTE: the conda locks are linux-64. On another arch, fall back to `env.yaml`.

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

# --- env 1: voxbind (py3.10). conda layer (byte-exact, no solver) then pip layer.
#     Copy just the locks first so these heavy layers cache across code changes. ---
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.conda-linux-64.lock /tmp/voxbind.conda-linux-64.lock
RUN micromamba create -y -n voxbind --file /tmp/voxbind.conda-linux-64.lock \
 && micromamba clean --all --yes
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxbind.pip.lock.txt /tmp/voxbind.pip.lock.txt
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba run -n voxbind pip install --no-cache-dir -r /tmp/voxbind.pip.lock.txt

# --- env 2: voxdock (py3.8) — paper-faithful Vina 1.2.2 docking stack. ---
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxdock.conda-linux-64.lock /tmp/voxdock.conda-linux-64.lock
RUN micromamba create -y -n voxdock --file /tmp/voxdock.conda-linux-64.lock \
 && micromamba clean --all --yes
COPY --chown=$MAMBA_USER:$MAMBA_USER env/voxdock.pip.lock.txt /tmp/voxdock.pip.lock.txt
RUN micromamba run -n voxdock pip install --no-cache-dir -r /tmp/voxdock.pip.lock.txt

# --- TargetDiff sibling (docking eval imports its utils.evaluation.* + full receptors) ---
RUN git clone --depth 1 https://github.com/guanjq/targetdiff /workspace/targetdiff

# --- project source (data/weights excluded via .dockerignore) + editable install. ---
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/VoxBind
RUN micromamba run -n voxbind pip install --no-cache-dir -e .

# Activate voxbind for every interactive shell / RUN.
ENV ENV_NAME=voxbind \
    MAMBA_DEFAULT_ENV=voxbind \
    HOME=/home/$MAMBA_USER \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    TARGETDIFF_ROOT=/workspace/targetdiff \
    FULL_RECEPTOR_ROOT=/workspace/targetdiff/data/test_set/test_set

CMD ["bash"]
