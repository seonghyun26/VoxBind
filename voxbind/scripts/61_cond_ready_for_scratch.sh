#!/usr/bin/env bash
# Condition gate for 99_chain.sh: is it time to start the from-scratch
# frozen-density-encoder VoxBind training?
#
# Exit 0 only when BOTH hold:
#   1. the VoxBind sampling evaluation has finished (no sample.py config_sample
#      processes left), and
#   2. MCP sampling (FuncBind sample_fb.py), if it is still running, is NOT
#      keeping the GPUs busy - mean per-process sm% over a short window stays
#      below MCP_MAX_SM.
#
# Usage:
#   bash scripts/99_chain.sh --gpus 0-3 --condition-script scripts/61_cond_ready_for_scratch.sh -- COMMAND
#
# Env knobs: MCP_MAX_SM (default 40), SAMPLES (pmon samples, default 10).
set -uo pipefail

MCP_MAX_SM=${MCP_MAX_SM:-40}
SAMPLES=${SAMPLES:-10}

if pgrep -f "config-name=config_sample" >/dev/null 2>&1; then
    echo "[61_cond] VoxBind evaluation still running - not ready"
    exit 1
fi

mapfile -t MCP_PIDS < <(pgrep -f "sample_fb.py" 2>/dev/null)

# Do not steal the GPUs from MCP chunks that are queued but have not started yet:
# the MCP launcher waits for free memory, so between "evaluation ended" and
# "first chunk running" there is a window where no sample_fb.py exists.
if (( ${#MCP_PIDS[@]} == 0 )); then
    if pgrep -f "run_mcpp_paper_run.sh" >/dev/null 2>&1; then
        echo "[61_cond] MCP launcher still queueing chunks - not ready"
        exit 1
    fi
    for d in /home1/irteam/funcbind/artifacts/reproduction/mcpp/paper_run/gpu*/; do
        [ -d "$d" ] || continue
        if [ ! -f "$d/exit_code" ]; then
            echo "[61_cond] MCP chunk $(basename "$d") launched but not finished - not ready"
            exit 1
        fi
    done
    echo "[61_cond] evaluation done, no MCP sampling running - ready"
    exit 0
fi

# Mean sm% across the MCP processes over SAMPLES one-second samples. Rows whose
# sm column is "-" mean the process held memory but issued no work: count as 0.
PIDS_CSV="$(IFS=,; echo "${MCP_PIDS[*]}")"
MEAN_SM="$(timeout $((SAMPLES + 20)) nvidia-smi pmon -c "$SAMPLES" -s u 2>/dev/null |
    awk -v pids="$PIDS_CSV" '
        BEGIN { n = split(pids, a, ","); for (i = 1; i <= n; i++) want[a[i]] = 1 }
        /^#/ { next }
        ($2 in want) { sm = ($4 == "-" ? 0 : $4); total += sm; count++ }
        END { if (count > 0) printf "%d", total / count; else print "0" }
    ')"
MEAN_SM=${MEAN_SM:-0}

if (( MEAN_SM < MCP_MAX_SM )); then
    echo "[61_cond] evaluation done, MCP sampling mean sm=${MEAN_SM}% < ${MCP_MAX_SM}% - ready"
    exit 0
fi

echo "[61_cond] MCP sampling busy (mean sm=${MEAN_SM}% >= ${MCP_MAX_SM}%) - not ready"
exit 1
