#!/usr/bin/env bash
# Auto-probe watcher for the coords+density campaign: waits for both encoders to finish
# (checkpoint_e0099), probes each on lp_edrscc_v2, and rebuilds results_h200.html.
# Runs independently of the orchestrator (which resumes the denoiser); the probes share
# GPU 0 with the resumed denoiser (small + brief). Launch detached:
#   setsid nohup bash scripts/autoprobe_coordsdens.sh > log/autoprobe_coordsdens.log 2>&1 &
set -uo pipefail
cd /home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin/python
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

RUNS=( atomblob_density_vit_mae_40m_plinder atomblob_density_channelvit_40m_plinder )
TAGS=( abd_vit abd_cvit )

# ── wait for both final checkpoints (max ~4h guard) ──────────────────────────
DEADLINE=$(( $(date +%s) + 4*3600 ))
for run in "${RUNS[@]}"; do
  log "waiting for exps/$run/checkpoint_e0099.pth.tar ..."
  while [ ! -f "exps/$run/checkpoint_e0099.pth.tar" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timed out waiting for $run"; exit 1; }
    sleep 60
  done
  log "  $run finished"
done
log "both encoders present; grace 60s for GPU settle"
sleep 60

# ── probe each (features → probe on lp_edrscc_v2) ─────────────────────────────
for i in 0 1; do
  run="${RUNS[$i]}"; tag="${TAGS[$i]}"
  log "FEATURES $run (atomblob_density, tag=$tag)"
  $PY -u dataset/01c_pdbbind_probe.py features --condition atomblob_density --exp_dir "exps/$run" \
    --epoch 99 --voxel_version v5 --tag "$tag" --device cuda --batch_size 24 2>&1 | grep -vE "encode |it/s|batch/s" | tail -8
  log "PROBE $run on lp_edrscc_v2"
  $PY -u dataset/01c_pdbbind_probe.py probe --conditions atomblob_density --exp_dir "exps/$run" \
    --epoch 99 --voxel_version v5 --feature_tag "$tag" --tag "$tag" \
    --split lp_edrscc_v2 --seeds 3 --device cuda 2>&1 | grep -iE "seed=|split sizes|wrote"
done

log "rebuilding results_h200.html"
$PY /home1/irteam/VoxBind/notebook/html/260625/build_results_h200.py
log "AUTOPROBE DONE — full 3x2 grid in notebook/html/260625/results_h200.html"
