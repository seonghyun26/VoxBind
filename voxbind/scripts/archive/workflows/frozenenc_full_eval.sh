#!/usr/bin/env bash
# Density-conditioned full eval for a FROZEN-ENC VoxBind. sample.py auto-skips no-density
# pockets for a with_density model, so this naturally covers the 79 x-ray-density
# CrossDocked test pockets (100 ligands each, 4-GPU split), then docks all of them —
# same 79-pocket set + identical Vina protocol as the frozen-enc ep350 eval.
#   bash scripts/archive/workflows/frozenenc_full_eval.sh <EXP_DIR> <SAMPLE_DIR> [n_samples]
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind; B=/opt/conda/envs/voxbind/bin; VOXDOCK=/opt/conda/envs/voxdock/bin; LOG=$VOX/log
EXP_DIR="$1"; SAMPLE_DIR="$2"; NS="${3:-100}"; TAG=$(basename "$SAMPLE_DIR")
export PATH=$B:${PATH}; export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
cd "$VOX" || exit 1; ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
mkdir -p "$SAMPLE_DIR"
log "SAMPLE frozen-enc $(basename $EXP_DIR) → $SAMPLE_DIR (density pockets × $NS, 4-GPU split)"
RANGES=("0 24" "25 49" "50 74" "75 99"); pids=()
for g in 0 1 2 3; do
  set -- ${RANGES[$g]}; S=$1; E=$2
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home1/irteam/VoxBind \
    "$B/python" sample.py --config-name=config_sample hydra.job.chdir=False dset=crossdocked_xray \
    dset.data_dir=$VOX/dataset/data dset.crops_dir=$VOX/dataset/data/xray_crops_aligned_v5 \
    dset.normalize=false dset.use_xray=true dset.pocket_radius=-1 dset.ligand_radius=0.5 \
    pretrained_path="$EXP_DIR" save_dir="$SAMPLE_DIR" \
    wjs.n_samples_per_pocket=$NS wjs.start=$S wjs.end=$E wjs.n_targets=100 \
    hydra.run.dir="$SAMPLE_DIR/_run_gpu${g}" > "$LOG/${TAG}_gpu${g}.log" 2>&1 &
  pids+=($!); log "  GPU$g pockets [$S,$E] (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; log "  sample pid $p exit=$?"; done
N=$(find "$SAMPLE_DIR" -maxdepth 2 -name samples.sdf 2>/dev/null | wc -l)
log "sampling done: $N density pockets with samples"
log "docking $N pockets (32-way niced, background CPU) → $SAMPLE_DIR/eval_docking_results.json"
setsid nohup nice -n 19 env PATH=$VOXDOCK:/usr/bin:/bin "$VOXDOCK/python" \
  exps/frozenenc_probes/run_docking_eval_parallel.py "$SAMPLE_DIR" \
  --out "$SAMPLE_DIR/eval_docking_results.json" --workers 32 --cpu 1 --exh 16 \
  > "$LOG/${TAG}_metrics.log" 2>&1 &
log "FROZENENC SAMPLING DONE ($TAG); docking launched on CPU (bg)."
