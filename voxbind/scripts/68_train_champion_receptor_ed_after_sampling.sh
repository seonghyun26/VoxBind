#!/usr/bin/env bash
# Wait for the current 4-GPU sampler, then train champion zero-init fusion on
# the largest successfully materialized receptor-ED subset. If that run exits
# non-zero, fall back to the latest completed recipe (efficient60M, reference
# ligand channels masked, cross-attention, scratch) with sigma changed to 1.0.
set -uo pipefail

ROOT=/home1/irteam/VoxBind/voxbind
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/exps/samples_scratch_crossattn_ep350}"
CROPS_DIR="${CROPS_DIR:-$ROOT/dataset/data/xray_crops_receptor_ed_v5}"
POLL="${POLL:-60}"
TIMEOUT="${TIMEOUT:-86400}"
PRIMARY_EXP="${PRIMARY_EXP:-voxbind_fusion_champion_reference_receptor_ed_zero_init_sig0.9}"
FALLBACK_EXP="${FALLBACK_EXP:-voxbind_frozen_efficient60m_scratch_crossattn_receptor_ed_sig1.0}"
PY=/opt/conda/envs/voxbind/bin/python
LOG="${LOG:-$ROOT/log/reference_receptor_ed_train_chain.log}"

mkdir -p "$ROOT/log"
cd "$ROOT" || exit 1
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [receptor-ed-chain] $*" | tee -a "$LOG"; }

say "waiting for four sampling chunks in $SAMPLE_DIR"
waited=0
while :; do
    n_done=$(find "$SAMPLE_DIR" -maxdepth 2 -path '*/_run_gpu*/exit_code' -type f 2>/dev/null | wc -l)
    if (( n_done >= 4 )); then
        break
    fi
    if (( waited >= TIMEOUT )); then
        say "timeout waiting for sampling after ${waited}s"
        exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done

for gpu in 0 1 2 3; do
    rc=$(cat "$SAMPLE_DIR/_run_gpu$gpu/exit_code")
    if [[ "$rc" != "0" ]]; then
        say "warning: sampling gpu$gpu exited $rc; training will still use the completed dataset"
    fi
done
say "sampling finished; waiting for receptor-ED dataset marker $CROPS_DIR/.complete"

waited=0
while [[ ! -f "$CROPS_DIR/.complete" ]]; do
    if (( waited >= TIMEOUT )); then
        say "timeout waiting for receptor-ED dataset after ${waited}s"
        exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done

TRAIN_AVAILABLE=$(
    "$PY" -c "import json; print(json.load(open('$CROPS_DIR/.complete'))['train']['available'])"
) || exit 1
TEST_AVAILABLE=$(
    "$PY" -c "import json; print(json.load(open('$CROPS_DIR/.complete'))['test']['available'])"
) || exit 1
if (( TRAIN_AVAILABLE <= 100 )); then
    say "invalid dataset size: train available=$TRAIN_AVAILABLE"
    exit 1
fi
SUBSET_N=$((TRAIN_AVAILABLE - 100))
say "dataset ready: train=$TRAIN_AVAILABLE ($SUBSET_N train + 100 val), test=$TEST_AVAILABLE"

run_primary() {
    say "launching $PRIMARY_EXP: champion frozen encoder, reference ligand + full receptor holo ED, default zero-init fusion, sigma=0.9"
    WARM_START="" EXP_NAME="$PRIMARY_EXP" NUM_EPOCHS=350 SIGMA=0.9 SEES_LIGAND=true MASK_LIGAND=false FUSION=default CROPS_DIR="$CROPS_DIR" SUBSET_N="$SUBSET_N" SUBSET_VAL_N=100 WANDB_TAGS="[voxbind,fusion,champion,reference_ligand,receptor_ed,full_holo_ed,zero_init,sigma0.9]" CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/67_train_fusion_champion_reference_4gpu.sh >>"$LOG" 2>&1
}

run_latest_fallback() {
    say "launching $FALLBACK_EXP: latest efficient60M reference-masked cross-attention scratch recipe, sigma=1.0"
    WARM_START="" EXP_NAME="$FALLBACK_EXP" NUM_EPOCHS=350 SIGMA=1.0 SEES_LIGAND=false MASK_LIGAND=false FUSION=cross_attn CROPS_DIR="$CROPS_DIR" SUBSET_N="$SUBSET_N" SUBSET_VAL_N=100 WANDB_TAGS="[voxbind,density_cond,frozen_encoder,efficient_60m,cross_attn,receptor_ed,reference_masked,sigma1.0]" CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/60_train_frozen_efficient60m_4gpu.sh >>"$LOG" 2>&1
}

run_primary
primary_rc=$?
if (( primary_rc == 0 )); then
    say "primary champion sigma=0.9 training completed successfully"
    exit 0
fi

say "primary training failed with exit=$primary_rc; falling back to latest completed recipe at sigma=1.0"
run_latest_fallback
fallback_rc=$?
say "latest-recipe fallback sigma=1.0 exit=$fallback_rc"
exit "$fallback_rc"
