#!/usr/bin/env bash
# rebuild_voxdock_env.sh
#   Rebuilds the isolated `voxdock` conda env used for Vina docking + chem metrics
#   (exps/frozenenc_probes/run_docking_eval{,_parallel}.py, which import TargetDiff's
#   utils.evaluation.{scoring_func,docking_vina} from /home1/irteam/TargetDIff).
#
#   WHY THIS EXISTS: /opt/conda/envs is EPHEMERAL on this box — a container restart
#   deletes it (see scripts/restore_voxbind_env.sh). `voxbind` is backed up to
#   data-vol1 and restores in ~30s; `voxdock` is NOT backed up, so it must be rebuilt
#   from source pins. Versions are taken verbatim from /home1/irteam/TargetDIff/
#   environment.yaml, which is what the working env was built from.
#
#   Deliberately installs ONLY the docking/chem subset — NOT torch/pyg/cuda from that
#   file. run_docking_eval.py needs rdkit + numpy + scipy + openbabel + easydict and
#   TargetDiff's pure-python eval modules; nothing imports torch.
#
#   No C compiler on this box (see [[voxbind-no-c-compiler]]) → every pin below must
#   resolve to a conda binary or a manylinux wheel. meeko is pinned to 0.1.dev3
#   because docking_vina.py uses the OLD OBMol-based API (preparator.prepare(OBMol),
#   write_pdbqt_file) which meeko >=0.5 removed.
#
#   After this, back it up so the next restart is a 30s restore, not a 10min rebuild:
#     tar -C /opt/conda/envs -cf - voxdock | zstd -T8 -6 \
#       > /home1/irteam/data-vol1/seonghyun/prj-sbdd/Voxbind/env-backups/voxdock-env.tar.zst
set -uo pipefail
ENV=voxdock
PREFIX=/opt/conda/envs/$ENV
TD=/home1/irteam/TargetDIff
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

[ -d "$TD" ] || { log "FATAL: TargetDiff not found at $TD"; exit 1; }
if [ -d "$PREFIX" ]; then log "$PREFIX already exists — remove it first to rebuild"; exit 1; fi

log "(1/3) conda create $ENV (python 3.8 + chem stack, conda-forge binaries)"
conda create -y -n "$ENV" -c conda-forge \
  python=3.8.16 numpy=1.24.3 scipy=1.10.1 rdkit=2022.03.2 openbabel=3.1.1 easydict pip \
  || { log "FATAL: conda create failed"; exit 1; }

log "(2/3) pip install docking stack (pins from $TD/environment.yaml)"
"$PREFIX/bin/pip" install --no-cache-dir \
  vina==1.2.2 meeko==0.1.dev3 pdb2pqr==3.6.1 propka==3.5.0 \
  mmcif-pdbx==2.0.1 docutils==0.17.1 \
  "git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git@aee55d50d5bdcfdbcd80220499df8cde2a8f4b2a" \
  || { log "FATAL: pip install failed"; exit 1; }

log "(3/3) smoke test"
PATH="$PREFIX/bin:$PATH" "$PREFIX/bin/python" - <<EOF || { log "FATAL: smoke test failed"; exit 1; }
import sys, os, shutil
sys.path.insert(0, "$TD")
import numpy as np
np.int = int          # vina 1.2.2 calls np.int (removed in numpy>=1.24) — same shim as run_docking_eval.py
from rdkit import Chem
from openbabel import pybel
from meeko import MoleculePreparation, obutils
from vina import Vina
import AutoDockTools
from utils.evaluation.scoring_func import get_chem
from utils.evaluation.docking_vina import VinaDockingTask
prep = os.path.join(AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py")
assert os.path.isfile(prep), "prepare_receptor4.py missing"
for b in ("pdb2pqr30", "obabel"):
    assert shutil.which(b), f"{b} not on PATH"
m = Chem.AddHs(Chem.MolFromSmiles("CCO")); Chem.AllChem.EmbedMolecule(m, randomSeed=1)
print("chem metrics on ethanol:", {k: round(v, 3) for k, v in get_chem(m).items() if isinstance(v, float)})
print("VOXDOCK SMOKE OK")
EOF
log "VOXDOCK ENV READY at $PREFIX"
