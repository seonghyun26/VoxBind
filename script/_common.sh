#!/usr/bin/env bash
# _common.sh — shared configuration + helpers, sourced by every 0N_*.sh script.
#
# Every path and knob below is overridable from the environment, so a coworker
# on a different box only edits their machine specifics (GPUs, data link, conda
# location) without touching the scripts. Source order inside each script:
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "$SCRIPT_DIR/_common.sh"
#
# Nothing here launches work; it only resolves paths and defines helpers.
set -uo pipefail

# --------------------------------------------------------------------------
# Repository layout (derived from this file's location; no absolute paths).
#   REPO_ROOT   .../VoxBind            (git checkout root)
#   CODE_ROOT   .../VoxBind/voxbind    (python package + train/sample entrypoints)
#   DATA_ROOT   .../VoxBind/voxbind/dataset/data
# --------------------------------------------------------------------------
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "$_COMMON_DIR/.." && pwd)}"
export CODE_ROOT="${CODE_ROOT:-$REPO_ROOT/voxbind}"
export DATA_ROOT="${DATA_ROOT:-$CODE_ROOT/dataset/data}"

# The repo's configs read these two (see env.local.sh / config cfg.yaml files).
export VOXBIND_ROOT="${VOXBIND_ROOT:-$CODE_ROOT}"
export VOXBIND_DATA_ROOT="${VOXBIND_DATA_ROOT:-$DATA_ROOT}"

# --------------------------------------------------------------------------
# Conda environments (all built by 00_setup_env.sh / the Dockerfile).
#   voxbind  GPU pipeline: torch cu118 + pyuul + hydra + rdkit + openbabel + gemmi
#   voxdock  Vina docking: python 3.8, vina 1.2.2, meeko 0.1.dev3, pdb2pqr, AutoDockTools
#   moleval  Pose quality: python 3.10, posecheck 1.3.1, prolif, posebusters
# --------------------------------------------------------------------------
export VOXBIND_ENV="${VOXBIND_ENV:-voxbind}"
export VOXDOCK_ENV="${VOXDOCK_ENV:-voxdock}"
export MOLEVAL_ENV="${MOLEVAL_ENV:-moleval}"

# --------------------------------------------------------------------------
# GPUs. GPUS is a comma list (e.g. "0,1,2,3"); NPROC is derived from it.
# Defaults to CUDA_VISIBLE_DEVICES if the caller exported it, else GPU 0.
# --------------------------------------------------------------------------
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export NPROC="$(awk -F, '{print NF}' <<< "$GPUS")"

# --------------------------------------------------------------------------
# Prepared-copy data source (see 01_download_data.sh).
#   DATA_RCLONE_REMOTE   rclone path to the prepared VoxBind data+weights folder
#   DATA_HTTP_URL        alternative: a single .tar / .tar.zst bundle over HTTP(S)
# Fill ONE of these in (or export it) before running 01_download_data.sh.
# --------------------------------------------------------------------------
export DATA_RCLONE_REMOTE="${DATA_RCLONE_REMOTE:-dropbox:/박성현/VoxBind/share}"
export DATA_HTTP_URL="${DATA_HTTP_URL:-}"   # e.g. https://.../voxbind_share.tar.zst

# --------------------------------------------------------------------------
# Evaluation externals (see 05_evaluate.sh).
#   TargetDiff must sit as a SIBLING of the VoxBind checkout (data.py resolves
#   VOXBIND.parent/targetdiff). FULL_RECEPTOR_ROOT points at its test_set.
# --------------------------------------------------------------------------
export TARGETDIFF_ROOT="${TARGETDIFF_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/targetdiff}"
export FULL_RECEPTOR_ROOT="${FULL_RECEPTOR_ROOT:-$TARGETDIFF_ROOT/data/test_set/test_set}"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
log()  { echo "[$(date '+%F %T')] $*"; }
die()  { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

# Resolve a conda-family launcher once (conda / mamba / micromamba). All three
# accept `run -n <env> <cmd...>`.
_resolve_conda() {
    if [ -n "${CONDA_LAUNCHER:-}" ]; then echo "$CONDA_LAUNCHER"; return; fi
    for c in micromamba mamba conda; do
        command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
    done
    return 1
}
export CONDA_LAUNCHER="${CONDA_LAUNCHER:-$(_resolve_conda || true)}"

# Run a command inside a named conda env. Usage: conda_run <env> <cmd...>
conda_run() {
    local env="$1"; shift
    [ -n "$CONDA_LAUNCHER" ] || die "no conda/mamba/micromamba on PATH; run 00_setup_env.sh or use the Docker image"
    if [ "$CONDA_LAUNCHER" = "conda" ]; then
        conda run --no-capture-output -n "$env" "$@"
    else
        "$CONDA_LAUNCHER" run -n "$env" "$@"
    fi
}

# Absolute python path for an env (used when a subprocess needs an interpreter,
# e.g. metrics.py's $MOLEVAL_PY worker).
env_python() {
    local env="$1"
    conda_run "$env" python -c 'import sys; print(sys.executable)' 2>/dev/null
}

# Assert an env exists early with a friendly message.
require_env() {
    local env="$1"
    env_python "$env" >/dev/null 2>&1 || \
        die "conda env '$env' not found — run: bash $_COMMON_DIR/00_setup_env.sh (or use the Docker image)"
}

require_data() {  # require_data <relative-path-under-DATA_ROOT> <human hint>
    [ -e "$DATA_ROOT/$1" ] || die "missing $DATA_ROOT/$1 — $2"
}

banner() { echo; echo "==== $* ===="; }
