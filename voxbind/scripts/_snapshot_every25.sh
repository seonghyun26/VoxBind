#!/usr/bin/env bash
# Archive the warm-start run's checkpoint every 25 epochs. train_ddp.py's
# save_checkpoint() writes ONE fixed filename (utils/base_utils.py:61-74) with no
# best-model tracking, so without this every epoch destroys the previous one and an
# overfit ep350 would be all that survives.
set -uo pipefail
D="$1"; PY=/opt/conda/envs/voxbind/bin/python
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib
mkdir -p "$D/snapshots"
while pgrep -f 'train_ddp.py --config-name' >/dev/null; do
  EP=$(grep -oE '>> epoch: [0-9]+' "$D/train_ddp.log" | tail -1 | grep -oE '[0-9]+$')
  if [ -n "$EP" ] && [ $((EP % 25)) -eq 0 ] && [ ! -f "$D/snapshots/checkpoint_e${EP}.pth.tar" ]; then
    prev=""; for _ in $(seq 20); do                       # wait for the write to settle
      cur=$(stat -c %Y-%s "$D/checkpoint.pth.tar" 2>/dev/null)
      [ "$cur" = "$prev" ] && break; prev=$cur; sleep 8
    done
    cp "$D/checkpoint.pth.tar" "$D/snapshots/.tmp_e${EP}" 2>/dev/null || { sleep 60; continue; }
    if "$PY" -c "import torch,sys; torch.load(sys.argv[1],map_location='cpu',weights_only=False)['epoch']" \
         "$D/snapshots/.tmp_e${EP}" >/dev/null 2>&1; then
      mv "$D/snapshots/.tmp_e${EP}" "$D/snapshots/checkpoint_e${EP}.pth.tar"
      echo "[$(date +%H:%M)] snapshot ep${EP} ok"
    else
      rm -f "$D/snapshots/.tmp_e${EP}"; echo "[$(date +%H:%M)] ep${EP} copy corrupt, retrying next cycle"
    fi
  fi
  sleep 120
done
echo "[$(date +%H:%M)] training ended — snapshot daemon exiting"
