#!/usr/bin/env bash
# 92_chain_gsplat_after_sampling_idle.sh
#   Start the CoDE + Gaussian-splat decoder run (91) ONLY IF the GPUs are still unused one
#   hour after the VoxBind+CoE n100 sampling finishes.
#
#   Why the hour: the MCP fine-tune is expected to take these GPUs back once sampling is
#   done (run from another session). If nothing has claimed them an hour later, this
#   curiosity run gets them; if anything has, this exits without touching it.
#
#   Detached on purpose (setsid nohup): a Claude session monitor dies with the session,
#   this does not. It is the ONLY thing that launches 91 -- nothing else should.
#
#   OOM fallback: 91 runs CoDE's bsz 32/rank. If the first attempt dies with a CUDA OOM,
#   it is relaunched once at bsz 16 x accum 2, which keeps CoDE's effective batch of 128.
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
cd "$ROOT" || exit 1

SAMPLE="$ROOT/exps/260908_fusion_default_cv2_scratch_8gpu/samples/samples_ep350_test79_n100"
WAIT_AFTER="${WAIT_AFTER:-3600}"
EXP_NAME="${EXP_NAME:-voxbind_code_gsplat_s4k2_sig0.9_$(date +%y%m%d)}"
LOGDIR="$ROOT/exps/frozenenc_probes/logs/gsplat"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/chain.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== gsplat chain armed: wait sampling -> +${WAIT_AFTER}s -> GPUs idle? -> 91 ==="
while [ "$(ls "$SAMPLE"/_run_gpu*/exit_code 2>/dev/null | wc -l)" -lt 4 ]; do sleep 120; done
say "sampling finished; waiting ${WAIT_AFTER}s before checking the GPUs"
sleep "$WAIT_AFTER"

busy=0
for i in 1 2 3 4 5; do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ]; then busy=1; say "GPU in use: $(echo "$apps" | tr '\n' ';')"; fi
    sleep 30
done
if [ "$busy" -ne 0 ]; then
    say "GPUS_BUSY — something claimed the GPUs (MCP fine-tune?); NOT starting the gsplat run"
    echo "GSPLAT_SKIPPED" >> "$LOG"
    exit 0
fi

launch() {   # $1 = bsz, $2 = accum, $3 = attempt tag
    local out="$LOGDIR/train_${3}.log"
    say "launching 91: EXP=$EXP_NAME bsz=$1 accum=$2 (log $out)"
    EXP_NAME="$EXP_NAME" BSZ="$1" ACCUM="$2" \
        setsid nohup bash scripts/91_train_code_gsplat_4gpu.sh >"$out" 2>&1 </dev/null &
    echo "$out"
}

# Decide OOM vs healthy on the log, within the first ~15 min (model build + warm start +
# the first training steps). A healthy run is left alone; this script then exits.
watch_first_steps() {   # $1 = log
    local log=$1 t=0
    while [ "$t" -lt 900 ]; do
        sleep 60; t=$((t + 60))
        if grep -qE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null; then
            echo OOM; return
        fi
        if grep -qE "Traceback|Error executing job|ChildFailedError" "$log" 2>/dev/null; then
            echo FAILED; return
        fi
        if ! pgrep -f "train_ddp.py --config-name config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1" >/dev/null; then
            echo EXITED; return
        fi
    done
    echo RUNNING
}

log1=$(launch 32 1 a1 | tail -1)
state=$(watch_first_steps "$log1")
say "attempt 1 (bsz 32): $state"
if [ "$state" = "OOM" ]; then
    pkill -f "91_train_code_gsplat_4gpu.sh" 2>/dev/null
    for p in $(pgrep -f "train_ddp.py --config-name config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1"); do
        kill "$p" 2>/dev/null
    done
    sleep 60
    log2=$(launch 16 2 a2 | tail -1)
    state=$(watch_first_steps "$log2")
    say "attempt 2 (bsz 16 x 2): $state"
fi
say "GSPLAT_CHAIN_DONE state=$state"
echo "GSPLAT_CHAIN_DONE state=$state" >> "$LOG"
