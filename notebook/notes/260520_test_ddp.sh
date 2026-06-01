#!/bin/bash
# DDP smoke test — runs 3 short trials on GPUs 4-7 to validate that
# voxbind/train_ddp.py works end-to-end. NOT a convergence test; just a
# functional check (does it launch? do all ranks contribute? does the loss
# stay finite? does a checkpoint get written?).
#
# Trials (all use dset=crossdocked, with_density=false, subset_n=8, 2 epochs):
#   T0  dp_1gpu : single GPU baseline   (train.py,     CUDA_VISIBLE_DEVICES=4)
#   T1  dp_w4   : DP across 4 GPUs       (train.py,     CUDA_VISIBLE_DEVICES=4-7)
#   T2  ddp_w4  : DDP across 4 ranks     (train_ddp.py, torchrun --nproc=4)
#
# Why subset_n=8 (not 2 as discussed): DistributedSampler with drop_last=True
# and world_size=4 needs >= 4 samples to produce any batch per rank. Picking 8
# gives a useful 2 batches/rank/epoch under DDP w=4 (bsz=2/rank) so we exercise
# the all-reduce path more than once per epoch.
#
# Outputs:
#   voxbind/log/smoke_ddp/<trial>/run.log       full stdout/stderr
#   voxbind/log/smoke_ddp/<trial>/results.json  parsed metrics + status
#   voxbind/exps/smoke_ddp_<trial>/             hydra output dir (checkpoints, cfg)

set -u  # error on unset vars; do NOT set -e so we can capture per-trial failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../VoxBind/notebook
REPO_ROOT="$(dirname "$SCRIPT_DIR")"         # .../VoxBind
VOXBIND_DIR="${REPO_ROOT}/voxbind"           # .../VoxBind/voxbind
cd "$VOXBIND_DIR"

# Use the voxbind conda env explicitly so we don't accidentally run on
# /compuworks/anaconda3/bin/python (which doesn't have the voxbind module).
PYBIN="${PYBIN:-/home/shpark/.conda/envs/voxbind/bin/python}"
TORCHRUN="${TORCHRUN:-/home/shpark/.conda/envs/voxbind/bin/torchrun}"
if [ ! -x "$PYBIN" ] || [ ! -x "$TORCHRUN" ]; then
    echo "ERROR: voxbind conda env not found at /home/shpark/.conda/envs/voxbind/"
    exit 1
fi

LOG_ROOT="${VOXBIND_DIR}/log/smoke_ddp"
mkdir -p "$LOG_ROOT"

DATA_DIR="${REPO_ROOT}/dataset/data"
if [ ! -f "${DATA_DIR}/data_train.pt" ]; then
    echo "ERROR: missing ${DATA_DIR}/data_train.pt"
    exit 1
fi

# Shared overrides for every trial. cfg.bsz is the only thing that differs:
# train.py interprets it as TOTAL (DP splits); train_ddp.py interprets it as
# PER-RANK (effective = bsz * world).
COMMON_OVERRIDES=(
    "dset=crossdocked"
    "dset.data_dir=${DATA_DIR}"
    "dset.subset_n=8"
    "model.with_density=false"
    # num_epochs=1 (NOT 2) sidesteps a latent bug in train.py:362 where
    # `model = model.module` is unconditionally applied when device_count > 1,
    # which crashes because the caller passes the already-unwrapped
    # model_ema.module. The bug triggers when `epoch > 0 and epoch == num_epochs-1`.
    # One epoch is enough for a smoke test (verifies init, fwd/bwd, all-reduce,
    # checkpoint write).
    "num_epochs=1"
    "num_workers=2"
    "wandb=false"
    "wjs.n_targets=0"
    "seed=42"
)

# Helper: snapshot nvidia-smi for the target GPUs before launching, so we have
# a record of contention at trial time.
gpu_snapshot() {
    local label="$1"
    local gpus="$2"  # comma-separated
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv --id="$gpus" 2>&1 \
        | sed "s/^/[${label}] /"
}

run_trial() {
    local trial="$1"        # e.g. dp_1gpu
    local mode="$2"         # dp | ddp
    local gpus="$3"         # comma-separated CUDA_VISIBLE_DEVICES
    local nproc="$4"        # only for ddp; ignored by dp
    local bsz="$5"
    local extra_overrides_str="$6"  # space-separated extra overrides

    local trial_dir="${LOG_ROOT}/${trial}"
    mkdir -p "$trial_dir"
    local log="${trial_dir}/run.log"
    local meta="${trial_dir}/meta.json"
    local exp_dir="exps/smoke_ddp_${trial}"
    local port=$((29510 + RANDOM % 100))

    # Skip if this trial already passed in a prior run (don't waste GPU time
    # re-validating something that already worked). Re-runs anything that
    # failed or never ran.
    if [ -f "$meta" ]; then
        local prev_exit
        prev_exit=$("$PYBIN" -c "import json; print(json.load(open('$meta')).get('exit_code', 1))" 2>/dev/null)
        if [ "$prev_exit" = "0" ]; then
            echo "====================================================================="
            echo "TRIAL: ${trial}  SKIP (already passed; meta.json has exit_code=0)"
            return 0
        fi
    fi

    echo "====================================================================="
    echo "TRIAL: ${trial}  (mode=${mode}  gpus=${gpus}  bsz=${bsz})"
    echo "log:    ${log}"
    echo "expdir: ${VOXBIND_DIR}/${exp_dir}"

    # Pre-flight GPU snapshot
    gpu_snapshot "pre" "$gpus" | tee "${trial_dir}/gpu_pre.txt"

    # Build the cfg.* override array. Read extras as space-split tokens.
    local overrides=("${COMMON_OVERRIDES[@]}" "bsz=${bsz}" "exp_name=smoke_ddp_${trial}" "output_dir=${exp_dir}")
    if [ -n "$extra_overrides_str" ]; then
        read -ra extras <<< "$extra_overrides_str"
        overrides+=("${extras[@]}")
    fi

    local t_start=$(date +%s)
    local exit_code=0

    if [ "$mode" = "dp" ]; then
        CUDA_VISIBLE_DEVICES="$gpus" \
            "$PYBIN" train.py "${overrides[@]}" >"$log" 2>&1
        exit_code=$?
    elif [ "$mode" = "ddp" ]; then
        # Force localhost for c10d rendezvous — `svr7` (this host) resolves to
        # an external IP (10.2.140.7) but the TCP store binds to localhost, so
        # the standalone hostname-based rendezvous deadlocks for 5 min.
        CUDA_VISIBLE_DEVICES="$gpus" \
        MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" \
            "$TORCHRUN" --nnodes=1 --nproc_per_node="$nproc" \
            --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:"$port" \
            train_ddp.py "${overrides[@]}" >"$log" 2>&1
        exit_code=$?
    else
        echo "ERROR: unknown mode '${mode}'"
        exit 1
    fi

    local t_end=$(date +%s)
    local elapsed=$((t_end - t_start))

    # Write meta JSON for the parser to consume
    python - "${trial}" "${mode}" "${gpus}" "${nproc:-1}" "${bsz}" "${exit_code}" "${elapsed}" "${log}" "${VOXBIND_DIR}/${exp_dir}" > "$meta" <<'PY'
import json, sys
keys = ["trial","mode","gpus","nproc","bsz","exit_code","wall_clock_s","log","exp_dir"]
vals = sys.argv[1:1+len(keys)]
# numeric fields
for k in ("nproc","bsz","exit_code","wall_clock_s"):
    i = keys.index(k); vals[i] = int(vals[i])
print(json.dumps(dict(zip(keys, vals)), indent=2))
PY

    # Post-flight GPU snapshot
    gpu_snapshot "post" "$gpus" | tee "${trial_dir}/gpu_post.txt"

    if [ $exit_code -eq 0 ]; then
        echo "TRIAL ${trial} OK   (${elapsed}s)"
    else
        echo "TRIAL ${trial} FAIL (exit=${exit_code}, ${elapsed}s)  see ${log}"
        # Don't bail — continue with the next trial so we capture all data points.
    fi
}

# -------- run the matrix --------

# T0-T2: original "does it launch?" smoke matrix (1 epoch).
run_trial "dp_1gpu" "dp"  "4"        "1" "2" ""
run_trial "dp_w4"   "dp"  "4,5,6,7"  "4" "8" ""
run_trial "ddp_w4"  "ddp" "4,5,6,7"  "4" "2" ""

# T3: convergence check — DDP w=4 on subset_n=64 for 5 epochs (40 optimizer
# steps). Verifies that the all-reduced gradients actually move the loss down.
run_trial "t3_converge" "ddp" "4,5,6,7" "4" "2" "dset.subset_n=64 num_epochs=5"

# T4: resume check — copy T3's exp dir, then launch DDP with cfg.resume so
# train_ddp.py loads the checkpoint, restores optimizer state, and continues
# training. cfg.num_epochs in the saved cfg.yaml is overwritten on the CLI.
T3_EXP_DIR="exps/smoke_ddp_t3_converge"
T4_EXP_DIR="exps/smoke_ddp_t4_resume"
T3_META="${LOG_ROOT}/t3_converge/meta.json"
T4_META="${LOG_ROOT}/t4_resume/meta.json"
T3_PASSED=0
if [ -f "$T3_META" ] && [ "$("$PYBIN" -c "import json; print(json.load(open('$T3_META')).get('exit_code', 1))" 2>/dev/null)" = "0" ]; then
    T3_PASSED=1
fi
T4_PASSED=0
if [ -f "$T4_META" ] && [ "$("$PYBIN" -c "import json; print(json.load(open('$T4_META')).get('exit_code', 1))" 2>/dev/null)" = "0" ]; then
    T4_PASSED=1
fi

T4_EXP_DIR_ABS="${VOXBIND_DIR}/${T4_EXP_DIR}"
if [ "$T4_PASSED" = "0" ] && [ "$T3_PASSED" = "1" ]; then
    if [ ! -f "${T3_EXP_DIR}/checkpoint.pth.tar" ]; then
        echo "WARN: t4_resume needs ${T3_EXP_DIR}/checkpoint.pth.tar but it's missing — skipping T4"
    else
        echo "==> Preparing T4: cp -r ${T3_EXP_DIR} ${T4_EXP_DIR}"
        rm -rf "$T4_EXP_DIR"
        cp -r "$T3_EXP_DIR" "$T4_EXP_DIR"
    fi
fi
# Absolute resume path so it works regardless of hydra.job.chdir.
run_trial "t4_resume" "ddp" "4,5,6,7" "4" "2" "dset.subset_n=64 num_epochs=2 resume=${T4_EXP_DIR_ABS}"

# T5: realistic-scale smoke — subset_n=100 (matches 01a_train_xray100.sh)
# with bsz=4/rank (matches BSZ=16 total there), 10 epochs. Specifically checks
# that DDP teardown is clean across multiple epochs at production-sized saves.
# Each epoch: 100/4=25 samples/rank, drop_last → 6 batches/rank/epoch =
# 6 optimizer steps × 10 epochs = 60 steps total per rank.
run_trial "t5_100samples" "ddp" "4,5,6,7" "4" "4" "dset.subset_n=100 num_epochs=10"

# -------- post-process: parse logs, write results.json + HTML --------

"$PYBIN" "${SCRIPT_DIR}/04a_test_ddp_to_html.py" --log-root "$LOG_ROOT" \
    --out-html "${REPO_ROOT}/notebook/html/ddp_smoke_test.html" \
    --out-json "${LOG_ROOT}/results.json"

echo "==> All trials done. Report: ${REPO_ROOT}/notebook/html/ddp_smoke_test.html"
