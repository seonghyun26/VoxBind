#!/bin/bash
# 37b_chain_v2probe_v3train_v3probe.sh — full v2/v3 chain after v2 e0099 lands.
#
#   1. wait for exps/260530_..._weighted_v2_pretrain/checkpoint_e0099.pth.tar
#   2. Phase 5 (v2): extract atomblob_merged_density features with the v2 encoder + v2 PDBbind density
#   3. Phase 6 (v2): probe results CSV  → probe_results_e99_v2.csv
#   4. launch scripts/38_train_..._v3.sh   (v3 retrain on GPUs 4-7)
#   5. wait for exps/260530_..._weighted_v3_pretrain/checkpoint_e0099.pth.tar
#   6. Phase 5 (v3): extract features with v3 encoder + v3 PDBbind density
#   7. Phase 6 (v3): probe results CSV  → probe_results_e99_v3.csv
#
# Logs:
#   voxbind/log/37b_chain_watcher.log              — stage-level events
#   voxbind/log/260530_..._v3_pretrain.log         — training writes its own
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
LOG=$VOX/log
W_LOG=$LOG/37b_chain_watcher.log

V2_EXP=260530_atomblob_merged_density_vit_mae_40m_weighted_v2_pretrain
V3_EXP=260530_atomblob_merged_density_vit_mae_40m_weighted_v3_pretrain
V2_CKPT=$VOX/exps/$V2_EXP/checkpoint_e0099.pth.tar
V3_CKPT=$VOX/exps/$V3_EXP/checkpoint_e0099.pth.tar

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"
echo "[$(ts)] 37b chain watcher started (PID $$)" >> "$W_LOG"

wait_for_ckpt() {
    local run="$1"; local ckpt="$2"
    echo "[$(ts)] waiting for $run  (gate: $ckpt)" >> "$W_LOG"
    while true; do
        [ -f "$ckpt" ] && { echo "[$(ts)] $run complete" >> "$W_LOG"; return 0; }
        if ! pgrep -af "$run" >/dev/null 2>&1; then
            echo "[$(ts)] ABORT: $run process gone but ckpt missing" >> "$W_LOG"
            return 1
        fi
        sleep 120
    done
}

run_probe() {
    # Phase 5 (merged_density only — the others have v1 features already) + Phase 6.
    local ver="$1"
    echo "[$(ts)] [$ver] Phase 5 extraction (atomblob_merged_density only)" >> "$W_LOG"
    CUDA_VISIBLE_DEVICES=5 $PY/python "$VOX/dataset/01c_pdbbind_probe.py" features \
        --condition atomblob_merged_density --voxel_version "$ver" \
        >> "$LOG/01e_features_${ver}.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(ts)] [$ver] ABORT: Phase 5 failed (rc=$RC); see $LOG/01e_features_${ver}.log" >> "$W_LOG"
        return $RC
    fi
    # Also ensure the atom-only conditions have symlinked features for this ver.
    for cond in atomblob atomblob_weighted; do
        CUDA_VISIBLE_DEVICES=5 $PY/python "$VOX/dataset/01c_pdbbind_probe.py" features \
            --condition "$cond" --voxel_version "$ver" \
            >> "$LOG/01e_features_${ver}.log" 2>&1
    done
    echo "[$(ts)] [$ver] Phase 6 probe" >> "$W_LOG"
    CUDA_VISIBLE_DEVICES=5 $PY/python "$VOX/dataset/01c_pdbbind_probe.py" probe \
        --voxel_version "$ver" \
        --conditions atomblob atomblob_weighted atomblob_merged_density \
        >> "$LOG/01f_probe_${ver}.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(ts)] [$ver] ABORT: Phase 6 failed (rc=$RC); see $LOG/01f_probe_${ver}.log" >> "$W_LOG"
        return $RC
    fi
    echo "[$(ts)] [$ver] probe done → $VOX/dataset/data/pdbbind/probe_results_e99_${ver}.csv" >> "$W_LOG"
    return 0
}

cd "$VOX" || exit 1

# ── Stage 1: wait for v2 e0099 ─────────────────────────────────────────────────
wait_for_ckpt "$V2_EXP" "$V2_CKPT" || exit 1
sleep 30                                    # DDP/wandb teardown

# ── Stage 2: probe v2 ──────────────────────────────────────────────────────────
run_probe v2 || exit 1

# ── Stage 3: launch v3 retrain (foreground; ~14-15 h) ──────────────────────────
echo "[$(ts)] launching v3 retrain (script 38)" >> "$W_LOG"
bash "$VOX/scripts/38_train_atomblob_merged_density_vit_mae_40m_weighted_v3.sh"
RC=$?
echo "[$(ts)] script 38 returned (exit $RC)" >> "$W_LOG"

# ── Stage 4: wait for v3 e0099 + probe ─────────────────────────────────────────
wait_for_ckpt "$V3_EXP" "$V3_CKPT" || exit 1
sleep 30
run_probe v3 || exit 1

echo "[$(ts)] chain watcher done" >> "$W_LOG"
