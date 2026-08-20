#!/usr/bin/env bash
# 62_eval_frozenenc_4gpu.sh — WJS-sample a frozen-density-encoder VoxBind checkpoint
# across all 4 H200s, 25 test targets per GPU (100 total).
#
#   EXP=voxbind_frozen_efficient60m_scratch_protfirst_sig0.9_20260802 \
#   OUT=samples_scratch_protfirst_ep350 bash scripts/62_eval_frozenenc_4gpu.sh
#
# The overrides below are copied verbatim from the 2026-08-01 warmstart evaluation
# (exps/samples_warmstart_efficient60m_ep349/_run_gpu0/.hydra/overrides.yaml) so the
# two runs are numerically comparable. Do not "tidy" them without re-checking that file.
#
# scripts/02_sample.sh does NOT work on this box: it hardcodes shpark's svr7 python
# (/home/shpark/.conda/envs/voxbind/bin/python) and expects crops under data/pretrain/,
# while this box keeps them at data/xray_crops_aligned_v5.
#
# Each chunk runs detached (setsid) and writes run.log + exit_code under $OUT/_run_gpuN.
set -uo pipefail

ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin/python

: "${EXP:?set EXP (experiment dir name under exps/)}"
: "${OUT:?set OUT (sample dir name under exps/)}"
SAMPLES=${SAMPLES:-100}      # wjs.n_samples_per_pocket
NTARGETS=${NTARGETS:-100}    # wjs.n_targets
PER_GPU=${PER_GPU:-25}
CHUNKS=${CHUNKS:-0,1,2,3}

# Density crops must match what the run TRAINED on. receptor-ED runs need
# xray_crops_receptor_ed_v5; its 79 shared test crops are hard-linked from
# aligned_v5 (byte-identical), so the 79-pocket comparison stays exact and the
# 13 extra pockets it covers just come along.
CROPS=${CROPS:-$ROOT/dataset/data/xray_crops_aligned_v5}

CKPT="$ROOT/exps/$EXP"
SAVE="$ROOT/exps/$OUT"
[ -f "$CKPT/checkpoint.pth.tar" ] || { echo "[62_eval] MISSING $CKPT/checkpoint.pth.tar"; exit 1; }
[ -x "$PY" ] || { echo "[62_eval] MISSING $PY — restore the voxbind env first"; exit 1; }

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1     # no C compiler on this box; inductor cannot build
# Keep GPU WJS busy while CPU-only bond reconstruction runs in spawned workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VOXBIND_RECON_WORKERS="${VOXBIND_RECON_WORKERS:-4}"
export VOXBIND_LOOKAHEAD_BATCHES="${VOXBIND_LOOKAHEAD_BATCHES:-2}"
export VOXBIND_GPU_PEAKS="${VOXBIND_GPU_PEAKS:-1}"
export VOXBIND_REFINE_BATCH="${VOXBIND_REFINE_BATCH:-25}"
mkdir -p "$SAVE"
cd "$ROOT" || exit 1

for g in ${CHUNKS//,/ }; do
    # wjs.end is INCLUSIVE (sample.py:56 skips only when pocket_id > end), so the last
    # index is lo+PER_GPU-1. Using lo+PER_GPU would re-sample each chunk boundary.
    lo=$((g * PER_GPU)); hi=$((lo + PER_GPU - 1))
    D="$SAVE/_run_gpu$g"
    mkdir -p "$D"; rm -f "$D/exit_code"
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=$g '$PY' sample.py --config-name=config_sample \
            hydra.job.chdir=False \
            dset=crossdocked_xray \
            dset.data_dir='$ROOT/dataset/data' \
            dset.crops_dir='$CROPS' \
            dset.normalize=false \
            dset.use_xray=true \
            dset.pocket_radius=-1 \
            dset.ligand_radius=0.5 \
            pretrained_path='$CKPT' \
            save_dir='$SAVE' \
            wjs.n_samples_per_pocket=$SAMPLES \
            wjs.start=$lo \
            wjs.end=$hi \
            wjs.n_targets=$NTARGETS \
            hydra.run.dir='$D' > '$D/run.log' 2>&1
        echo \$? > '$D/exit_code'
    " </dev/null >"$D/launch.log" 2>&1 &
    echo "[62_eval] gpu$g: targets $lo-$hi -> $D (pid $!)"
done

echo "[62_eval] all chunks launched; samples land in $SAVE/target_XX/"
