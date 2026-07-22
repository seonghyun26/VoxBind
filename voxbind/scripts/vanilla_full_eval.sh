#!/usr/bin/env bash
# Full generation + docking eval for a VANILLA (no-density) VoxBind denoiser, matched to
# the frozen-enc eval. Samples 100 ligands for ALL 100 CrossDocked test pockets (4-GPU
# split; vanilla has no density branch so it doesn't skip pockets), then docks ONLY the
# same 79 density pockets the frozen-enc used (symlink filter) so the numbers + native/
# cross split are directly comparable.
#   bash scripts/vanilla_full_eval.sh <EXP_DIR> <SAMPLE_DIR> [n_samples]
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
VOXDOCK=/opt/conda/envs/voxdock/bin
LOG=$VOX/log
EXP_DIR="$1"; SAMPLE_DIR="$2"; NS="${3:-100}"
FROZ_EVAL=$VOX/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350
DOCK_DIR="$SAMPLE_DIR/_dock79"
TAG=$(basename "$SAMPLE_DIR")
export PATH=$B:${PATH}
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }

mkdir -p "$SAMPLE_DIR"
log "SAMPLE vanilla $(basename $EXP_DIR) → $SAMPLE_DIR (100 pockets × $NS ligands, 4-GPU split)"
RANGES=("0 24" "25 49" "50 74" "75 99"); pids=()
for g in 0 1 2 3; do
  set -- ${RANGES[$g]}; S=$1; E=$2
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home1/irteam/VoxBind \
    "$B/python" sample.py --config-name=config_sample hydra.job.chdir=False \
    pretrained_path="$EXP_DIR" save_dir="$SAMPLE_DIR" \
    wjs.n_samples_per_pocket=$NS wjs.start=$S wjs.end=$E wjs.n_targets=100 \
    hydra.run.dir="$SAMPLE_DIR/_run_gpu${g}" > "$LOG/${TAG}_gpu${g}.log" 2>&1 &
  pids+=($!); log "  GPU$g pockets [$S,$E] (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; log "  sample pid $p exit=$?"; done
log "sampling done: $(find $SAMPLE_DIR -maxdepth 2 -name samples.sdf 2>/dev/null|wc -l) pockets with samples"

# dock ONLY the 79 pockets the frozen-enc used (matched comparison)
mkdir -p "$DOCK_DIR"
n=0
for t in $(ls -d $FROZ_EVAL/target_* 2>/dev/null | xargs -n1 basename); do
  [ -d "$SAMPLE_DIR/$t" ] && { ln -sfn "$SAMPLE_DIR/$t" "$DOCK_DIR/$t"; n=$((n+1)); }
done
log "docking $n matched pockets (32-way, niced, background CPU) → $SAMPLE_DIR/eval_docking_results.json"
setsid nohup nice -n 19 env PATH=$VOXDOCK:/usr/bin:/bin "$VOXDOCK/python" \
  exps/frozenenc_probes/run_docking_eval_parallel.py "$DOCK_DIR" \
  --out "$SAMPLE_DIR/eval_docking_results.json" --workers 32 --cpu 1 --exh 16 \
  > "$LOG/${TAG}_metrics.log" 2>&1 &
log "VANILLA SAMPLING DONE ($TAG); docking launched on CPU (bg). GPUs free for the next stage."
