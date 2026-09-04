#!/usr/bin/env bash
# 81_chain_cv2_scratch_after_sampling.sh
#   Wait for the running CDG_v2 scratch chain to finish SAMPLING, then start the C_v2
#   (coords-only) scratch arm on the freed GPUs.
#
#   WHY WAIT ON SAMPLING, NOT ON THE WHOLE CHAIN: chain 80 is training (GPU) -> sampling
#   (GPU) -> Vina/PoseCheck (CPU, ~2.5 h). The GPUs are free the moment sampling ends, so
#   gating on the eval would idle 8 GPUs for hours — which already cost ~2 days once.
#
#   WHAT C_v2 IS FOR: it is the matched control for the density question. C_v2 is the same
#   100M ChannelViT trunk pretrained on the same PLINDER v2 data, but on coords ONLY
#   (input_mode=atomblob, n_in=11, groups [7,4]) — no rho, no ||grad rho||. Running it
#   through the identical v4 token fusion holds the fusion mechanism, the frozen-encoder
#   capacity, the 78.5k subset, the schedule and the seed fixed, so the gap against the
#   CDG_v2 arm is attributable to the DENSITY CHANNELS rather than to "some frozen encoder
#   helps" or to subset fine-tuning.
#
#   Both arms are from scratch at 350 epochs, so neither carries warm-start drift.
#
#   Geometry is derived from the encoder's own cfg.yaml by 75 (n_in=11, groups [7,4]);
#   models/voxbind.py gained a coords-only branch for this — without it the 1-channel map
#   reaches an 11-channel patch embed and dies with
#   "split_with_sizes expects split_sizes to sum exactly to 1 ... got [7, 4]".
#
#   Env knobs: EXP_NAME, TOTAL_EPOCHS, WAIT_EXP, POLL.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

WAIT_EXP="${WAIT_EXP:-260902_fusion_v4_cdgv2_scratch_8gpu}"
WAIT_OUT="${WAIT_OUT:-samples_ep350_test79}"
EXP_NAME="${EXP_NAME:-260905_fusion_v4_cv2_scratch_8gpu}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-350}"
ENCODER="${ENCODER:-$ROOT/voxbind/model_zoo/C_v2/checkpoint_e0020.pth.tar}"
POLL="${POLL:-300}"

WAIT_SAVE="$ROOT/voxbind/exps/$WAIT_EXP/samples/$WAIT_OUT"
LOG="$ROOT/voxbind/logs/${EXP_NAME}.chain.log"
mkdir -p "$ROOT/voxbind/logs"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -f "$ENCODER" ] || { say "MISSING encoder $ENCODER"; exit 1; }
say "=== C_v2 scratch arm queued; waiting for $WAIT_EXP to finish sampling ==="

while :; do
    n=$(ls -d "$WAIT_SAVE"/target_* 2>/dev/null | wc -l)
    train_up=$(ps -eo cmd | grep -c "[t]rain_ddp\.py")
    samp_up=$(ps -eo cmd | grep -c "[s]ample\.py")
    # GPUs are free only when BOTH the trainer and every sampling shard are gone. The
    # target-dir count alone is not enough: 72 shards write target dirs as they go, so 79
    # can appear while later shards are still running.
    if [ "$train_up" -eq 0 ] && [ "$samp_up" -eq 0 ] && [ "$n" -ge 79 ]; then
        say "sampling complete ($n target dirs, no GPU jobs) — starting C_v2"
        break
    fi
    say "waiting: targets=$n train_procs=$train_up sample_procs=$samp_up"
    sleep "$POLL"
done

export ENCODER
say "launching C_v2 scratch chain: exp=$EXP_NAME epochs=$TOTAL_EPOCHS encoder=$ENCODER"
EXP_NAME="$EXP_NAME" TOTAL_EPOCHS="$TOTAL_EPOCHS" ENCODER="$ENCODER" \
    bash scripts/80_chain_scratch_v4_full.sh >>"$LOG" 2>&1
say "=== C_v2 chain exited $? ==="
