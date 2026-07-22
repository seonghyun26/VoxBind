#!/usr/bin/env bash
# Post-cap pipeline for the frozen-enc VoxBind gen run:
#   1) wait until the frozen-enc checkpoint reaches epoch >= CAP (350),
#   2) snapshot the ep350 ckpt, suppress the sig=1.0 fallback, stop training,
#   3) FULL generation eval: 100 ligands for each density-available test pocket,
#      split across the 4 GPUs (sample.py is single-GPU → 4 parallel slices),
#   4) launch a FRESH vanilla sig=0.9 denoiser from scratch (GPUs) AND the Vina/
#      chem metrics (CPU) IN PARALLEL — docking doesn't need GPUs, so the denoiser
#      trains while docking runs.
# Replaces chain_cap350.sh (kill that first). Kills chain_frozenenc_restart.sh at cap
# so its sig=1.0 resume does NOT fire.
#   setsid nohup bash scripts/chain_frozenenc_eval_then_sig09.sh > log/chain_frozenenc_eval_then_sig09.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
VOXDOCK=/opt/conda/envs/voxdock/bin
LOG=$VOX/log
FROZ_EXP=voxbind_frozenenc_atomblob7_v2p1_sig0.9
FROZ_DIR=$VOX/exps/$FROZ_EXP
CKPT=$FROZ_DIR/checkpoint.pth.tar
CAP=350
SNAP=$VOX/exps/frozenenc_atomblob7_v2p1_ep350_snap
SAMPLE_DIR=$FROZ_DIR/samples/full_eval_ep350
SIG09_EXP=exp_sig0.9_v2
SIG09_DIR=$VOX/exps/$SIG09_EXP
LOCK=$LOG/chain_frozenenc_eval_then_sig09.lock
export PATH=$B:${PATH}
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another eval-then-sig09 chain running — exit"; exit 1; }
# retire the old cap-watcher at startup so only THIS chain handles the cap (no race).
# (this script's process cmdline is its own path, so this pattern won't self-match)
pkill -9 -f '[c]hain_cap350.sh' && log "retired old cap-watcher (chain_cap350.sh)" || log "no old cap-watcher"
ckpt_epoch(){ "$B/python" -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null; }
drain(){ for _ in $(seq 1 48); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-999999}" -lt 3000 ] && break; sleep 5; done; log "GPU drained (${U:-?} MiB)"; }

# ── 1) wait for the cap ──────────────────────────────────────────────────────
DEADLINE=$(( $(date +%s) + 4*24*3600 ))
log "waiting for frozen-enc to reach epoch $CAP ..."
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout waiting for cap"; exit 1; }
  ep=$(ckpt_epoch "$CKPT")
  { [ -n "$ep" ] && [ "$ep" -ge "$CAP" ]; } 2>/dev/null && { log "frozen-enc reached epoch $ep >= $CAP"; break; }
  log "epoch=$ep (<$CAP) — waiting ..."; sleep 600
done

# ── 2) snapshot ep350 ckpt (retry vs live-write), stop training, suppress sig1.0 ──
mkdir -p "$SNAP"; cp "$FROZ_DIR/cfg.yaml" "$SNAP/cfg.yaml"
for t in 1 2 3 4 5; do
  cp "$CKPT" "$SNAP/checkpoint.pth.tar"
  sep=$(ckpt_epoch "$SNAP/checkpoint.pth.tar")
  { [ -n "$sep" ] && [ "$sep" -ge "$CAP" ]; } 2>/dev/null && { log "snapshot OK (epoch=$sep)"; break; }
  log "snapshot retry $t (epoch=$sep)"; sleep 20
done
pkill -9 -f '[c]hain_frozenenc_restart.sh' && log "restart-chain killed (sig1.0 suppressed)" || log "no restart-chain"
pkill -9 -f '[c]hain_cap350.sh' && log "old cap-watcher killed" || true
sleep 2
for p in $(pgrep -f 'python3.*train_ddp.py'); do kill -TERM "$p" 2>/dev/null; done
sleep 15
for p in $(pgrep -f 'python3.*train_ddp.py'); do kill -9 "$p" 2>/dev/null; done
drain

# ── 3) FULL generation eval — 100 ligands/pocket, 4-GPU pocket slices ─────────
mkdir -p "$SAMPLE_DIR"
log "GEN eval → $SAMPLE_DIR (100 ligands × density pockets, 4-GPU; ep350 snapshot)"
RANGES=("0 24" "25 49" "50 74" "75 99"); pids=()
for g in 0 1 2 3; do
  set -- ${RANGES[$g]}; S=$1; E=$2
  GLOG="$LOG/full_eval_ep350_gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home1/irteam/VoxBind \
    "$B/python" sample.py --config-name=config_sample \
    hydra.job.chdir=False dset=crossdocked_xray \
    dset.data_dir=$VOX/dataset/data dset.crops_dir=$VOX/dataset/data/xray_crops_aligned_v5 \
    dset.normalize=false dset.use_xray=true dset.pocket_radius=-1 dset.ligand_radius=0.5 \
    pretrained_path="$SNAP" save_dir="$SAMPLE_DIR" \
    wjs.n_samples_per_pocket=100 wjs.start=$S wjs.end=$E wjs.n_targets=100 \
    hydra.run.dir="$SAMPLE_DIR/_run_gpu${g}" > "$GLOG" 2>&1 &
  pids+=($!); log "  GPU$g pockets [$S,$E] → $GLOG (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; log "  sampling pid $p exit=$?"; done
NGEN=$(find "$SAMPLE_DIR" -name samples.sdf 2>/dev/null | wc -l)
log "GEN eval done: $NGEN pockets with samples in $SAMPLE_DIR"
drain

# ── 4a) fresh vanilla sig=0.9 denoiser from scratch (GPUs) ────────────────────
DENLOG="$LOG/${SIG09_EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH fresh sig=0.9 denoiser (from scratch) exp=$SIG09_EXP → $DENLOG"
setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 TORCHDYNAMO_DISABLE=1 \
  LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib \
  "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_ddp.py" smooth_sigma=0.9 bsz=32 accum_steps=1 num_workers=12 \
  +prefetch_factor=16 num_epochs=400 wjs.n_targets=0 \
  exp_name="$SIG09_EXP" output_dir="$SIG09_DIR" hydra.run.dir="$SIG09_DIR" > "$DENLOG" 2>&1 &
log "sig=0.9 denoiser launched (log: $DENLOG)."

# ── 4b) metrics (Vina + QED/SA/diversity) — PARALLEL on CPU, alongside the denoiser ─
# ~7.9k docks fanned across cores (64 targets × vina cpu=1 ≈ 64 cores), leaving the rest
# for the denoiser's dataloaders. CPU-only → never touches the GPUs.
MLOG="$LOG/full_eval_ep350_metrics.log"
log "LAUNCH parallel docking metrics (voxdock env, 32-way CPU, nice) → $MLOG"
# MUST put voxdock/bin on PATH (VinaDockingTask shells out to pdb2pqr30/obabel) AND
# nice+cap workers to 32 — 64-way starved the denoiser's dataloaders (GPU util → 15%).
setsid nohup nice -n 19 env PATH=/opt/conda/envs/voxdock/bin:/usr/bin:/bin \
  "$VOXDOCK/python" exps/frozenenc_probes/run_docking_eval_parallel.py "$SAMPLE_DIR" \
  --out "$SAMPLE_DIR/eval_docking_results.json" --workers 32 --cpu 1 --exh 16 > "$MLOG" 2>&1 &
log "metrics launched (32-way CPU, niced). CHAIN DONE — denoiser on GPUs + docking on CPU."
