#!/usr/bin/env bash
# 72_sample_8gpu.sh — WJS-sample a VoxBind checkpoint over the full 100-pocket
# CrossDocked test split, sharded across all 8 GPUs.
#
#   EXP=260827_voxbind_base_8gpu OUT=samples_ep350_test100 bash scripts/72_sample_8gpu.sh
#
# sample.py is single-GPU and walks the sampling loader in order, so the split is
# by pocket index: wjs.start/wjs.end carve 0..99 into 8 contiguous chunks, one per
# GPU, all writing target_XX/ into the SAME save_dir. wjs.end is INCLUSIVE
# (sample.py breaks only when pocket_id > end), so the last index is lo+n-1 —
# using lo+n would re-sample every chunk boundary.
#
# Every wjs value below is the SHIPPED default from configs/wjs/sampling.yaml
# (unmodified since the voxbind-oss commit 863a94a): split=test, n_targets=100,
# n_samples_per_pocket=10, chain_init=denovo, warmup=400, steps=100, max_steps=100,
# mask_pocket=1. Only wjs.start/wjs.end are overridden, and those exist in the
# config precisely to select a pocket range.
#
# n_targets=100 is safe with sharding: sample.py's second break is
# `pocket_id == n_targets`, and the test split only has ids 0..99, so it never
# fires — wjs.end alone bounds each chunk.
#
# Caveat: cfg.seed (1234) is applied per process, so 8 shards each start the same
# RNG stream instead of one continuous stream. Sharded sampling is therefore not
# bit-identical to a single-process run; the protocol and per-pocket settings are.
#
# Env knobs: EXP, OUT, SAMPLES, SPLIT, NTARGETS, GPUS, N_POCKETS.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
PY="$HOME/miniforge3/envs/voxbind/bin/python"
cd "$ROOT/voxbind" || exit 1

# XRAY_CROPS — REQUIRED for a density-conditioned checkpoint (with_density=true).
# sample.py does `cfg = OmegaConf.merge(cfg_model, cfg)`, i.e. the SAMPLE-time config
# wins over the checkpoint's. config_sample.yaml defaults to `dset: crossdocked`, so a
# density model silently loses dset_name=crossdocked_xray, the batch arrives with no
# "xray_density" key, and sample.py's `if with_density and density is None: skipped`
# drops EVERY pocket — a run that exits 0 having written nothing. Setting this points
# the sampling loader back at the x-ray dataset and the v5 crops.
#   XRAY_CROPS=dataset/data/pretrain/xray_crops_aligned_v5
# EXPECT_TARGETS — density conditioning samples only pockets that HAVE a map (79 of the
# 100 test pockets for v5), so the final count check must expect 79, not 100.
XRAY_CROPS="${XRAY_CROPS:-}"

: "${EXP:?set EXP (experiment dir name under exps/)}"
OUT="${OUT:-samples_test100}"
SAMPLES="${SAMPLES:-10}"       # wjs.n_samples_per_pocket (shipped default)
SPLIT="${SPLIT:-test}"
N_POCKETS="${N_POCKETS:-100}"  # test split size
NTARGETS="${NTARGETS:-100}"    # shipped default; safe with sharding, see above
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EXPECT_TARGETS="${EXPECT_TARGETS:-$N_POCKETS}"

XRAY_ARGS=()
if [ -n "$XRAY_CROPS" ]; then
    [ -d "$XRAY_CROPS/$SPLIT" ] || { echo "[72_sample] MISSING $XRAY_CROPS/$SPLIT"; exit 1; }
    XRAY_ARGS=(
        "dset=crossdocked_xray"
        "dset.crops_dir=$XRAY_CROPS"
        "dset.normalize=false"
        "dset.use_xray=true"
        "dset.pocket_radius=-1"
        "dset.ligand_radius=0.5"
    )
fi

CKPT="$ROOT/voxbind/exps/$EXP"
SAVE="$CKPT/samples/$OUT"
[ -f "$CKPT/checkpoint.pth.tar" ] || { echo "[72_sample] MISSING $CKPT/checkpoint.pth.tar"; exit 1; }
[ -x "$PY" ] || { echo "[72_sample] MISSING $PY"; exit 1; }

# The env's editable install points at a deleted checkout; see 70_train's header.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NG=${#GPU_LIST[@]}
mkdir -p "$SAVE"

# Contiguous, balanced chunks: the first (N_POCKETS % NG) chunks take one extra.
base=$((N_POCKETS / NG)); rem=$((N_POCKETS % NG)); lo=0
echo "[72_sample] exp=$EXP split=$SPLIT pockets=$N_POCKETS samples/pocket=$SAMPLES gpus=$NG"
echo "[72_sample] save_dir=$SAVE"

for i in "${!GPU_LIST[@]}"; do
    g="${GPU_LIST[$i]}"
    n=$base; [ "$i" -lt "$rem" ] && n=$((base + 1))
    hi=$((lo + n - 1))
    D="$SAVE/_run_gpu$g"
    mkdir -p "$D"; rm -f "$D/exit_code"
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=$g '$PY' sample.py --config-name=config_sample \
            hydra.job.chdir=False \
            pretrained_path='$CKPT' \
            save_dir='$SAVE' \
            out_dir='$OUT' \
            wjs.split=$SPLIT \
            wjs.n_samples_per_pocket=$SAMPLES \
            wjs.n_targets=$NTARGETS \
            wjs.start=$lo \
            wjs.end=$hi \
            ${XRAY_ARGS[*]:-} \
            hydra.run.dir='$D' > '$D/run.log' 2>&1
        echo \$? > '$D/exit_code'
    " </dev/null >"$D/launch.log" 2>&1 &
    echo "[72_sample]   gpu$g: pockets $lo-$hi ($n)"
    lo=$((hi + 1))
done

echo "[72_sample] waiting for $NG chunks..."
while :; do
    done_n=$(ls "$SAVE"/_run_gpu*/exit_code 2>/dev/null | wc -l)
    [ "$done_n" -ge "$NG" ] && break
    sleep 60
done

fail=0
for f in "$SAVE"/_run_gpu*/exit_code; do
    rc=$(cat "$f"); [ "$rc" -ne 0 ] && { echo "[72_sample] FAILED: $f rc=$rc"; fail=1; }
done
n_targets_done=$(ls -d "$SAVE"/target_* 2>/dev/null | wc -l)
echo "[72_sample] chunks finished; target dirs = $n_targets_done / $EXPECT_TARGETS"
[ "$fail" -ne 0 ] && exit 1
[ "$n_targets_done" -ne "$EXPECT_TARGETS" ] && { echo "[72_sample] WARNING: expected $EXPECT_TARGETS target dirs"; exit 1; }
echo "[72_sample] done -> $SAVE"
