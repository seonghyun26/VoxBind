#!/bin/bash
# 12_sample_baseline_10k_test.sh — test-split sampling for the 10k BASELINE arm.
#
# Completes the 3-way 10k test-split A/B. 09_sample_10k_test_ab.sh sampled the
# density & noise arms; this samples the baseline (260518_voxbind_10k_baseline,
# with_density=false). Waits for the 10k baseline TRAINING (08_*.sh, GPUs 4-7)
# to finish, then samples it on the CrossDocked test split on GPU 7.
#
# The baseline is with_density=false, so sample.py samples ALL 100 test pockets
# (it only auto-skips no-density pockets for density-conditioned models). After
# sampling, the 8 pockets without x-ray density are trimmed, leaving exactly the
# same 92 density-having pockets the density/noise arms used (aligned eval).
#
# Out: exps/260518_voxbind_10k_baseline/samples/res_ep99_test/
#
# Launch detached:
#   nohup bash scripts/12_sample_baseline_10k_test.sh \
#       > log/260519_sample_baseline_10k_test.log 2>&1 &
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
BASE_EXP=260518_voxbind_10k_baseline
GPU=${GPU:-7}
TRAIN_LOG=$LOG/260518_train_10k_ab.log
ORCH_PID=$(cat "$LOG/260518_train_10k_ab.pid" 2>/dev/null || true)
OUT=$VOX/exps/$BASE_EXP/samples/res_ep99_test
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo $$ > "$LOG/260519_sample_baseline_10k_test.pid"
echo "[base-test $(ts)] waiting for the 10k baseline training to finish ..."

# wait for 08_*.sh to log the baseline arm complete
while ! grep -q "ARM 'baseline' finished" "$TRAIN_LOG" 2>/dev/null; do
    if [ -n "$ORCH_PID" ] && ! kill -0 "$ORCH_PID" 2>/dev/null; then
        sleep 15
        grep -q "ARM 'baseline' finished" "$TRAIN_LOG" 2>/dev/null && break
        echo "[base-test $(ts)] ERROR: 08 orchestrator (pid $ORCH_PID) exited but the"
        echo "                  'ARM baseline finished' marker is absent — aborting."
        exit 1
    fi
    sleep 120
done
echo "[base-test $(ts)] 10k baseline training finished."
sleep 30   # let GPUs free / final checkpoint settle

if [ ! -f "$VOX/exps/$BASE_EXP/checkpoint.pth.tar" ]; then
    echo "[base-test $(ts)] ERROR: $VOX/exps/$BASE_EXP/checkpoint.pth.tar missing"
    exit 1
fi

echo "[base-test $(ts)] sampling baseline on the TEST split, GPU $GPU"
echo "[base-test $(ts)]   out=$OUT"
echo "[base-test $(ts)]   log=$LOG/260519_sample_10k_baseline_test.log"
CUDA_VISIBLE_DEVICES=$GPU $PY $VOX/sample.py \
    pretrained_path=$VOX/exps/$BASE_EXP \
    dset=crossdocked_xray \
    dset.data_dir=$DATA \
    dset.crops_dir=$DATA/xray_crops \
    wjs.split=test wjs.start=0 wjs.end=99 wjs.n_targets=100 \
    wjs.n_samples_per_pocket=10 \
    out_dir=res_ep99_test \
    save_dir=$OUT \
    > "$LOG/260519_sample_10k_baseline_test.log" 2>&1
rc=$?
echo "[base-test $(ts)] sampling exit rc=$rc"

# Trim to the 92 pockets that have x-ray density, so the baseline output aligns
# 1:1 (by target index) with the density/noise arms. Idempotent + defensive.
$PY - "$OUT" "$DATA/xray_crops/test_available.npy" <<'PYEOF'
import sys, os, shutil
import numpy as np
out, avail_path = sys.argv[1], sys.argv[2]
avail = np.load(avail_path)
targets = sorted(d for d in os.listdir(out) if d.startswith('target_'))
n_dens = int(avail.sum())
print(f"[trim] baseline sampled {len(targets)} target dirs; "
      f"test_available: {n_dens}/{len(avail)} test pockets have x-ray density")
if len(targets) != len(avail):
    print(f"[trim] WARNING: sampled count ({len(targets)}) != n test pockets "
          f"({len(avail)}) -> NOT trimming; leaving output as-is for inspection")
    sys.exit(0)
removed = 0
for k, ok in enumerate(avail):
    if not ok:
        p = os.path.join(out, f"target_{k:02d}")
        if os.path.isdir(p):
            shutil.rmtree(p); removed += 1
kept = sorted(d for d in os.listdir(out) if d.startswith('target_'))
print(f"[trim] removed {removed} no-density target dirs; kept {len(kept)} "
      f"(expected {n_dens})")
PYEOF
echo "[base-test $(ts)] DONE -> $OUT"
