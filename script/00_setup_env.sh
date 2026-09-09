#!/usr/bin/env bash
# 00_setup_env.sh — environment preparation for the full VoxBind pipeline.
#
# Builds the three conda envs the pipeline needs and clones the TargetDiff
# sibling used by the evaluation stage. Idempotent: an env / clone that already
# exists is left alone (pass --force to rebuild a specific env).
#
#   bash script/00_setup_env.sh                 # build everything that's missing
#   bash script/00_setup_env.sh voxbind         # just the GPU env
#   bash script/00_setup_env.sh voxdock moleval # just the eval envs
#   FORCE=1 bash script/00_setup_env.sh voxdock # rebuild an env from scratch
#
# If you are using the Docker image (script/Dockerfile) you do NOT need this —
# the image already contains all three envs + TargetDiff. Use this for a native
# (bare-metal / cluster) install.
#
# Envs
#   voxbind  GPU pipeline (env.yaml + `pip install -e .`)
#   voxdock  Vina 1.2.2 docking stack (python 3.8; pins are load-bearing for the
#            paper's absolute affinities — do not bump)
#   moleval  PoseCheck 1.3.1 + PoseBusters + ProLIF pose-quality stack (python 3.10)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

FORCE="${FORCE:-}"
TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=(voxbind voxdock moleval targetdiff)

[ -n "$CONDA_LAUNCHER" ] || die "no conda/mamba/micromamba found. Install Miniforge first: https://github.com/conda-forge/miniforge"
log "using conda launcher: $CONDA_LAUNCHER"

_env_exists() { env_python "$1" >/dev/null 2>&1; }

_maybe_drop() {  # remove env if FORCE set
    local env="$1"
    if _env_exists "$env" && [ -n "$FORCE" ]; then
        log "FORCE: removing existing env '$env'"
        "$CONDA_LAUNCHER" env remove -y -n "$env" 2>/dev/null || "$CONDA_LAUNCHER" remove -y -n "$env" --all
    fi
}

# Install an env from a conda-lock.yml (restores conda + pip in one shot).
# Finds conda-lock on PATH, else a local `_condalock` helper env, else instructs.
_install_from_condalock() {  # _install_from_condalock <env> <lockfile>
    local env="$1" lock="$2" cl=""
    if command -v conda-lock >/dev/null 2>&1; then
        cl="conda-lock"
    elif env_python _condalock >/dev/null 2>&1; then
        cl="$CONDA_LAUNCHER run -n _condalock conda-lock"
    else
        die "conda-lock not found — install it first: pip install conda-lock   (or: $CONDA_LAUNCHER create -y -n _condalock -c conda-forge conda-lock)"
    fi
    local solver=--conda
    case "$CONDA_LAUNCHER" in *micromamba) solver=--micromamba ;; *mamba) solver=--mamba ;; esac
    # shellcheck disable=SC2086
    $cl install $solver -n "$env" "$lock" || die "$env conda-lock install failed"
}

build_voxbind() {
    _maybe_drop voxbind
    if _env_exists voxbind; then log "env 'voxbind' already present — skipping (FORCE=1 to rebuild)"; return; fi
    local lock="$REPO_ROOT/env/voxbind.conda-lock.yml"
    if [ -f "$lock" ]; then
        banner "building 'voxbind' from conda-lock.yml (py3.10, vina 1.2.7)"
        _install_from_condalock voxbind "$lock"
    else
        banner "building 'voxbind' from env.yaml (no conda-lock.yml found — looser pin)"
        "$CONDA_LAUNCHER" env create -y -n voxbind -f "$REPO_ROOT/env.yaml" \
            || "$CONDA_LAUNCHER" create -y -n voxbind -f "$REPO_ROOT/env.yaml" \
            || die "voxbind env create failed"
    fi
    log "installing VoxBind (editable) into 'voxbind'"
    ( cd "$REPO_ROOT" && conda_run voxbind pip install --no-cache-dir -e . ) || die "pip install -e . failed"
    log "voxbind ready"
}

build_voxdock() {
    _maybe_drop voxdock
    if _env_exists voxdock; then log "env 'voxdock' already present — skipping"; return; fi
    local lock="$REPO_ROOT/env/voxdock.conda-lock.yml"
    if [ -f "$lock" ]; then
        banner "building 'voxdock' from conda-lock.yml (py3.8, vina 1.2.2)"
        _install_from_condalock voxdock "$lock"
    else
        banner "building 'voxdock' from recipe (Vina 1.2.2 docking stack, python 3.8)"
        # meeko 0.1.dev3 keeps the OBMol API docking_vina.py depends on. vina==1.2.2 is
        # the exact build every reported VoxBind affinity uses (no py3.10 wheel → py3.8).
        "$CONDA_LAUNCHER" create -y -n voxdock -c conda-forge \
            python=3.8.16 numpy=1.24.3 scipy=1.10.1 rdkit=2022.03.2 openbabel=3.1.1 \
            easydict pip || die "voxdock conda create failed"
        conda_run voxdock pip install --no-cache-dir \
            vina==1.2.2 meeko==0.1.dev3 pdb2pqr==3.6.1 propka==3.5.0 \
            mmcif-pdbx==2.0.1 docutils==0.17.1 \
            "git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git@aee55d50d5bdcfdbcd80220499df8cde2a8f4b2a" \
            || die "voxdock pip install failed"
    fi
    log "voxdock ready"
}

build_moleval() {
    _maybe_drop moleval
    if _env_exists moleval; then log "env 'moleval' already present — skipping"; return; fi
    banner "building 'moleval' (PoseCheck 1.3.1 + PoseBusters + ProLIF, python 3.10)"
    "$CONDA_LAUNCHER" create -y -n moleval python=3.10 || die "moleval conda create failed"
    conda_run moleval pip install --no-cache-dir \
        posebusters 'pandas>=2.2.3' prolif datamol hydride biopython rdkit \
        || die "moleval pose stack pip install failed"
    # PIN posecheck: upstream silently redefined strain energy; 1.3.1 is the
    # definition all reported numbers use.
    conda_run moleval pip install --no-cache-dir --no-deps 'posecheck==1.3.1' \
        || die "moleval posecheck pip install failed"
    # reduce = the protonation binary PoseCheck shells out to.
    "$CONDA_LAUNCHER" install -y -n moleval -c conda-forge reduce || log "WARN: could not install 'reduce' (posecheck protonation may fail)"
    log "moleval ready"
}

clone_targetdiff() {
    if [ -d "$TARGETDIFF_ROOT/utils/evaluation" ]; then
        log "TargetDiff already present at $TARGETDIFF_ROOT — skipping clone"
    else
        banner "cloning TargetDiff (sibling of the VoxBind checkout)"
        git clone https://github.com/guanjq/targetdiff "$TARGETDIFF_ROOT" \
            || die "git clone targetdiff failed"
    fi
    if [ ! -d "$FULL_RECEPTOR_ROOT" ]; then
        log "NOTE: full-receptor test_set not found at:"
        log "        $FULL_RECEPTOR_ROOT"
        log "      It ships in TargetDiff's data release (test_set.zip) and is also"
        log "      folded into the prepared-copy download (01_download_data.sh)."
        log "      The docking/pose eval needs it; chem-only metrics do not."
    fi
}

for t in "${TARGETS[@]}"; do
    case "$t" in
        voxbind)    build_voxbind ;;
        voxdock)    build_voxdock ;;
        moleval)    build_moleval ;;
        targetdiff) clone_targetdiff ;;
        *) die "unknown target '$t' (use: voxbind | voxdock | moleval | targetdiff)" ;;
    esac
done

banner "environment summary"
for e in voxbind voxdock moleval; do
    if _env_exists "$e"; then printf "  %-8s OK  (%s)\n" "$e" "$(env_python "$e")"; else printf "  %-8s MISSING\n" "$e"; fi
done
[ -d "$TARGETDIFF_ROOT/utils/evaluation" ] && echo "  targetdiff OK  ($TARGETDIFF_ROOT)" || echo "  targetdiff MISSING ($TARGETDIFF_ROOT)"
echo
log "done. Next: bash script/01_download_data.sh"
