#!/usr/bin/env bash
# 04_sample.sh — walk-jump sample ligands from a trained VoxBind checkpoint,
# sharded across the available GPUs over the 100-pocket CrossDocked test split.
#
#   EXP=voxbind_density_fusion bash script/04_sample.sh
#   EXP=voxbind_density_fusion OUT=samples_ep100 SAMPLES=10 GPUS=0,1,2,3 bash script/04_sample.sh
#
# For a DENSITY-CONDITIONED checkpoint (model.with_density=true) the sampler MUST
# be pointed back at the x-ray dataset + crops, otherwise every pocket without a
# map is silently skipped and the run writes nothing. This script auto-detects
# density conditioning from the checkpoint's cfg.yaml and wires the crops in.
# Density models sample only pockets that HAVE a map (~79 of the 100 test pockets
# for v5), so a lower target count than 100 is expected and fine.
#
# sample.py is single-GPU and walks the loader in pocket order; we carve the
# split into contiguous per-GPU chunks (wjs.start/end, end INCLUSIVE) all writing
# target_XX/ into one save_dir.
#
# Key overrides: EXP, OUT, SAMPLES, SPLIT, N_POCKETS, GPUS, XRAY_CROPS.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

: "${EXP:?set EXP (experiment dir name under voxbind/exps/, e.g. voxbind_density_fusion)}"
OUT="${OUT:-samples_test100}"
SAMPLES="${SAMPLES:-10}"          # wjs.n_samples_per_pocket (paper default)
SPLIT="${SPLIT:-test}"
N_POCKETS="${N_POCKETS:-100}"
NTARGETS="${NTARGETS:-100}"
XRAY_CROPS="${XRAY_CROPS:-$DATA_ROOT/pretrain/xray_crops_aligned_v5}"

require_env "$VOXBIND_ENV"
CKPT_DIR="$CODE_ROOT/exps/$EXP"
[ -f "$CKPT_DIR/checkpoint.pth.tar" ] || die "no checkpoint at $CKPT_DIR/checkpoint.pth.tar (train it with 03_train.sh)"
[ -f "$CKPT_DIR/cfg.yaml" ]           || die "no cfg.yaml at $CKPT_DIR (not a valid exp dir)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Detect density conditioning from the checkpoint's own config.
WITH_DENSITY="$(conda_run "$VOXBIND_ENV" python - "$CKPT_DIR/cfg.yaml" <<'PY'
import sys, yaml
m = (yaml.safe_load(open(sys.argv[1])) or {})
print(str(bool((m.get("model") or {}).get("with_density", False))).lower())
PY
)"

XRAY_ARGS=()
if [ "$WITH_DENSITY" = "true" ]; then
    [ -d "$XRAY_CROPS/$SPLIT" ] || die "density model needs crops, missing $XRAY_CROPS/$SPLIT (rerun 01/02)"
    XRAY_ARGS=( dset=crossdocked_xray "dset.crops_dir=$XRAY_CROPS" dset.normalize=false
                dset.use_xray=true dset.pocket_radius=-1 dset.ligand_radius=0.5 )
    log "density-conditioned checkpoint -> sampling only map-backed pockets from $XRAY_CROPS"
else
    log "coords-only checkpoint -> sampling all $N_POCKETS pockets"
fi

# Resolve the env interpreter once — cleaner than nesting `conda run` inside the
# backgrounded per-GPU subshells (which mangles quoting).
VOXBIND_PY_BIN="$(env_python "$VOXBIND_ENV")"
[ -x "$VOXBIND_PY_BIN" ] || die "could not resolve python for env '$VOXBIND_ENV'"

SAVE="$CKPT_DIR/samples/$OUT"
mkdir -p "$SAVE"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NG=${#GPU_LIST[@]}
banner "sampling exp=$EXP  split=$SPLIT  pockets=$N_POCKETS  samples/pocket=$SAMPLES  gpus=$NG"
log "save_dir=$SAVE"

# Contiguous balanced chunks; first (N_POCKETS % NG) chunks take one extra pocket.
base=$((N_POCKETS / NG)); rem=$((N_POCKETS % NG)); lo=0
for i in "${!GPU_LIST[@]}"; do
    g="${GPU_LIST[$i]}"
    n=$base; [ "$i" -lt "$rem" ] && n=$((base + 1))
    [ "$n" -eq 0 ] && continue
    hi=$((lo + n - 1))
    D="$SAVE/_run_gpu$g"; mkdir -p "$D"; rm -f "$D/exit_code"
    setsid nohup bash -c "
        cd '$CODE_ROOT'
        CUDA_VISIBLE_DEVICES=$g '$VOXBIND_PY_BIN' sample.py --config-name=config_sample hydra.job.chdir=False \
            pretrained_path='$CKPT_DIR' save_dir='$SAVE' out_dir='$OUT' \
            wjs.split=$SPLIT wjs.n_samples_per_pocket=$SAMPLES wjs.n_targets=$NTARGETS \
            wjs.start=$lo wjs.end=$hi ${XRAY_ARGS[*]:-} hydra.run.dir='$D' > '$D/run.log' 2>&1
        echo \$? > '$D/exit_code'
    " </dev/null >"$D/launch.log" 2>&1 &
    log "  gpu$g: pockets $lo-$hi ($n)"
    lo=$((hi + 1))
done

log "waiting for $NG chunk(s)..."
while :; do
    done_n=$(ls "$SAVE"/_run_gpu*/exit_code 2>/dev/null | wc -l)
    [ "$done_n" -ge "$NG" ] && break
    sleep 30
done

fail=0
for f in "$SAVE"/_run_gpu*/exit_code; do
    rc=$(cat "$f"); [ "$rc" -ne 0 ] && { log "FAILED chunk $(dirname "$f" | xargs basename) rc=$rc (see run.log)"; fail=1; }
done
n_done=$(ls -d "$SAVE"/target_* 2>/dev/null | wc -l)
banner "sampling finished: $n_done target dir(s) under $SAVE"
[ "$fail" -ne 0 ] && die "one or more chunks failed"
log "done. Next:  SAMPLE_DIR=$SAVE bash script/05_evaluate.sh"
