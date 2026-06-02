#!/bin/bash
# 08_train_10k_ab.sh — 10k-sample retraining of the X-ray density A/B test.
#
# Scales the 100-sample pilot (260514/260515/260516) up to subset_n=10000.
#
#   STEP 1  Regenerate noise crops for the 10k subset ->
#           dataset/data/xray_crops_noise_10k/  (00c_make_noise_density.py,
#           n_subset = subset_n + subset_val_n). The pilot's xray_crops_noise/
#           is a separate dir and is left untouched.
#   STEP 2  Train 3 arms SEQUENTIALLY on GPUs 4-7 (DDP, 4 ranks, effective
#           batch 16), 100 epochs each — hyperparameters identical to the pilot
#           except subset_n (100 -> 10000) and num_epochs (8000 -> 100):
#             260518_voxbind_10k_density  : with_density=true,  crops=xray_crops
#             260518_voxbind_10k_noise    : with_density=true,  crops=xray_crops_noise_10k
#             260518_voxbind_10k_baseline : with_density=false, crops=xray_crops
#           All 3 arms train on the identical 10k pockets (pool[:10000]) and the
#           identical 100-pocket val carve (pool[10000:10100]).
#
# Launch detached:
#   cd voxbind
#   nohup bash scripts/08_train_10k_ab.sh > log/260518_train_10k_ab.log 2>&1 &
#
# Env vars (optional): GPUS (4,5,6,7), NPROC (4), EPOCHS (100),
#                      SUBSET_N (10000), SUBSET_VAL_N (100).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOXBIND_DIR="$(dirname "$SCRIPT_DIR")"
cd "$VOXBIND_DIR"

LOG_DIR="${VOXBIND_DIR}/log"
DATA_DIR="${VOXBIND_DIR}/dataset/data"
EXPS="${VOXBIND_DIR}/exps"

GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-100}"
SUBSET_N="${SUBSET_N:-10000}"
SUBSET_VAL_N="${SUBSET_VAL_N:-100}"

XRAY_CROPS="${DATA_DIR}/xray_crops"
NOISE_CROPS="${DATA_DIR}/xray_crops_noise_10k"

VOXBIND_ENV_BIN="${VOXBIND_ENV_BIN:-/home/shpark/.conda/envs/voxbind/bin}"
export PATH="${VOXBIND_ENV_BIN}:${PATH}"

log()  { echo "[train10k $(date '+%F %T')] $*"; }
fail() { log "FATAL: $*"; exit 1; }

echo "$$" > "${LOG_DIR}/260518_train_10k_ab.pid"
log "10k A/B retraining starting (pid $$)"
log "  GPUs=$GPUS  nproc=$NPROC  epochs=$EPOCHS  subset_n=$SUBSET_N  subset_val_n=$SUBSET_VAL_N"

# --- preflight ---------------------------------------------------------
command -v torchrun >/dev/null 2>&1 || fail "torchrun not found (PATH=$PATH)"
command -v python   >/dev/null 2>&1 || fail "python not found"
[ -f "${VOXBIND_DIR}/train_ddp.py" ]              || fail "train_ddp.py missing"
[ -f "${VOXBIND_DIR}/dataset/00c_make_noise_density.py" ] || fail "00c script missing"
[ -d "${XRAY_CROPS}/train" ]                      || fail "missing ${XRAY_CROPS}/train"
[ -f "${DATA_DIR}/data_train.pt" ]                || fail "missing data_train.pt"

# ======================================================================
# STEP 1 — regenerate noise crops for the 10k subset
# ======================================================================
N_CROPS=$((SUBSET_N + SUBSET_VAL_N))
NOISE_GEN_LOG="${LOG_DIR}/260518_make_noise_density_10k.log"
have_crops="$(ls "${NOISE_CROPS}/train" 2>/dev/null | wc -l)"
if [ "$have_crops" -ge "$N_CROPS" ]; then
    log "STEP 1: ${NOISE_CROPS}/train already has $have_crops crops (>= $N_CROPS) — skipping 00c"
else
    log "STEP 1: generating $N_CROPS noise crops -> ${NOISE_CROPS}  (log: $NOISE_GEN_LOG)"
    python dataset/00c_make_noise_density.py \
        --real_crops_dir "${XRAY_CROPS}" \
        --out_dir "${NOISE_CROPS}" \
        --n_subset "$N_CROPS" \
        --seed 42 \
        > "$NOISE_GEN_LOG" 2>&1 \
        || fail "00c noise-crop generation failed — see $NOISE_GEN_LOG"
fi
have_crops="$(ls "${NOISE_CROPS}/train" 2>/dev/null | wc -l)"
[ "$have_crops" -ge "$N_CROPS" ] || fail "noise crops < $N_CROPS after STEP 1 (have $have_crops)"
log "STEP 1 done: ${NOISE_CROPS}/train has $have_crops crops"

# ======================================================================
# STEP 2 — train the 3 arms sequentially on GPUs $GPUS
# ======================================================================
train_arm() {
    local arm="$1" crops="$2" with_density="$3"
    local exp_name="260518_voxbind_10k_${arm}"
    local out_dir="${EXPS}/${exp_name}"
    local arm_log="${LOG_DIR}/${exp_name}.log"
    log "==> ARM '${arm}': with_density=${with_density}  crops=$(basename "$crops")"
    log "    out=${out_dir}  log=${arm_log}"
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="$NPROC" \
        train_ddp.py \
        dset=crossdocked_xray \
        dset.data_dir="${DATA_DIR}" \
        dset.crops_dir="${crops}" \
        dset.subset_n="${SUBSET_N}" \
        dset.subset_xray_only=true \
        dset.subset_val_n="${SUBSET_VAL_N}" \
        dset.use_xray=true \
        num_epochs="${EPOCHS}" \
        bsz=4 \
        seed=42 \
        wjs.n_targets=0 \
        model.with_density="${with_density}" \
        exp_name="${exp_name}" \
        output_dir="${out_dir}" \
        > "${arm_log}" 2>&1
    local rc=$?
    log "==> ARM '${arm}' finished (rc=${rc})"
    return $rc
}

train_arm density  "${XRAY_CROPS}"  true  ; RC_D=$?
train_arm noise    "${NOISE_CROPS}" true  ; RC_N=$?
train_arm baseline "${XRAY_CROPS}"  false ; RC_B=$?

# --- summary -----------------------------------------------------------
log "=============================================================="
log "10k A/B retraining done — rc(density=$RC_D  noise=$RC_N  baseline=$RC_B)"
for arm in density noise baseline; do
    d="${EXPS}/260518_voxbind_10k_${arm}"
    ep="$(grep -oE 'epoch: [0-9]+' "${d}/train_ddp.log" 2>/dev/null | tail -1)"
    log "  260518_voxbind_10k_${arm}: last ${ep:-epoch ?}"
done
log "=============================================================="
