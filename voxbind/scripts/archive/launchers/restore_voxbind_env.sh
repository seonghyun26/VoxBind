#!/usr/bin/env bash
# Restore the voxbind conda env from the persistent backup on data-vol1 after a container
# restart wiped /opt/conda/envs/voxbind (the env dir is ephemeral on this box).
# This is the FAST path (~30s extract) vs scripts/archive/launchers/rebuild_voxbind_env.sh (~10min, needs pip/net).
# Works because the backup is restored to the SAME absolute prefix it was packed from
# (/opt/conda/envs/voxbind) -> no conda-pack prefix rewriting needed. No C compiler required:
# pure extract. If the backup is missing/corrupt, fall back to rebuild_voxbind_env.sh.
set -uo pipefail
BACKUP=/home1/irteam/data-vol1/seonghyun/prj-sbdd/Voxbind/env-backups/voxbind-env.tar.zst
ENVDIR=/opt/conda/envs/voxbind
ZSTD=/opt/conda/bin/zstd
PY=$ENVDIR/bin/python
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

[ -f "$BACKUP" ] || { echo "[$(ts)] NO BACKUP at $BACKUP -> run scripts/archive/launchers/rebuild_voxbind_env.sh instead"; exit 1; }

echo "[$(ts)] (1/3) integrity-test backup"
"$ZSTD" -t "$BACKUP" || { echo "FAILED: backup corrupt -> use rebuild_voxbind_env.sh"; exit 1; }

if [ -e "$ENVDIR" ]; then
  echo "[$(ts)] env already exists at $ENVDIR -> moving aside to ${ENVDIR}.old"
  rm -rf "${ENVDIR}.old"; mv "$ENVDIR" "${ENVDIR}.old"
fi

echo "[$(ts)] (2/3) extract $BACKUP -> /opt/conda/envs/"
"$ZSTD" -dc "$BACKUP" | tar -C /opt/conda/envs -xf - || { echo "FAILED: extract"; exit 1; }

echo "[$(ts)] (3/3) smoke test"
export LD_LIBRARY_PATH=$ENVDIR/lib:${LD_LIBRARY_PATH:-}
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
from voxbind.dataset import create_dataloaders
from voxbind.models import create_model
print("VOXBIND TRAIN IMPORTS OK")
PY
RC=$?
[ "$RC" -eq 0 ] && echo "[$(ts)] ENV RESTORE COMPLETE (rm -rf ${ENVDIR}.old when satisfied)" \
               || echo "[$(ts)] RESTORE smoke test FAILED (rc=$RC)"
exit $RC
