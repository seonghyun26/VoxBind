#!/usr/bin/env bash
# rebuild_moleval_env.sh
#   Rebuilds the isolated `moleval` conda env used for PoseCheck + PoseBusters
#   structural-quality eval (notebook/webapp/pose_eval.py, invoked as a subprocess
#   worker by notebook/webapp/metrics.py --pose ...).
#
#   WHY THIS EXISTS: /opt/conda/envs is EPHEMERAL on this box — a container restart
#   deletes it. Unlike `voxbind` (backed up to data-vol1) and `voxdock` (rebuilt by
#   rebuild_voxdock_env.sh), `moleval` had NO build script, so a restart left
#   `metrics.py --pose` silently recording pose errors.
#
#   The pins below are the recipe metrics.py itself prints when the toolchain is
#   missing (see the `pose_eval_available()` warning block in _cli()) — i.e. what
#   the working env was built from. prolif/MDAnalysis cannot live in the `voxbind`
#   env without risking the training deps, hence the separate env.
#
#   No C compiler on this box (see [[voxbind-no-c-compiler]]) → every package must
#   resolve to a conda binary or a manylinux wheel. python=3.10 is what the recipe
#   pins; hydride/MDAnalysis publish cp310 manylinux wheels.
#
#   After this, back it up so the next restart is a restore, not a rebuild:
#     tar -C /opt/conda/envs -cf - moleval | zstd -T8 -6 \
#       > /home1/irteam/data-vol1/seonghyun/prj-sbdd/Voxbind/env-backups/moleval-env.tar.zst
set -uo pipefail
ENV=moleval
PREFIX=/opt/conda/envs/$ENV
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

if [ -d "$PREFIX" ]; then log "$PREFIX already exists — remove it first to rebuild"; exit 1; fi

log "(1/4) conda create $ENV (python 3.10)"
conda create -y -n "$ENV" python=3.10 \
  || { log "FATAL: conda create failed"; exit 1; }

log "(2/4) pip install pose stack"
conda run -n "$ENV" pip install --no-cache-dir \
  posebusters 'pandas>=2.2.3' prolif datamol hydride biopython rdkit \
  || { log "FATAL: pip install (pose stack) failed"; exit 1; }

log "(3/4) pip install posecheck (--no-deps: its deps are pinned above)"
conda run -n "$ENV" pip install --no-cache-dir --no-deps posecheck \
  || { log "FATAL: pip install posecheck failed"; exit 1; }

log "(4/4) conda install reduce (the protonation binary PoseCheck shells out to)"
conda install -y -n "$ENV" -c bioconda -c conda-forge \
  reduce seaborn xorg-libxrender xorg-libxext \
  || { log "FATAL: conda install reduce failed"; exit 1; }

log "smoke test"
PATH="$PREFIX/bin:$PATH" "$PREFIX/bin/python" - <<'EOF' || { log "FATAL: smoke test failed"; exit 1; }
import shutil, sys
from posecheck import PoseCheck
from posebusters import PoseBusters
import prolif, hydride, rdkit
print("posecheck  : import OK")
print("posebusters: import OK")
print("prolif     :", prolif.__version__)
print("rdkit      :", rdkit.__version__)
# PoseCheck shells out to `reduce` for pocket protonation; without it every
# target records a protein-level error instead of clashes/strain.
assert shutil.which("reduce"), "reduce binary not on PATH"
print("reduce     :", shutil.which("reduce"))
PoseCheck()
print("PoseCheck(): constructed OK")
EOF

log "DONE — moleval env ready at $PREFIX"
