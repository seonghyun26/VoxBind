#!/bin/bash
# 00_watch_n_launch.sh — wait for a training checkpoint to land, then run the
# next command. Generalizes the one-off watcher scripts 19a/32a/33a/37a into a
# single composable primitive.
#
# Behaviour:
#   1. (pre-gate)  block until every --wait PATH exists. If a --watch PATTERN is
#      given and that process disappears before the gate file appears, ABORT
#      (a crashed run must not trigger downstream training on a partial ckpt).
#   2. (grace)     sleep --grace seconds for DDP/wandb/GPU teardown.
#   3. (launch)    run the COMMAND after "--" (blocks until it exits).
#   4. (post-gate) if --expect PATH is given, require it after COMMAND exits,
#      else exit non-zero.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/00_watch_n_launch.sh [options] -- COMMAND [ARGS...]
#
# Options:
#   --wait PATH       gate file to wait for (repeatable for multiple gate files)
#   --watch PATTERN   pgrep -f pattern; abort if it dies before the gate appears
#   --interval SEC    poll interval                       (default 120)
#   --grace SEC       sleep after the gate, before COMMAND (default 30)
#   --expect PATH     require this file after COMMAND exits (repeatable)
#   --log FILE        watcher log (default log/00_watch_n_launch.log)
#   --dry-run         print what it would do and exit
#
# ── Examples (replacing the old watchers) ─────────────────────────────────────
#   # 19a/33a: wait for a pretrain ckpt, then launch downstream training
#   bash scripts/00_watch_n_launch.sh \
#       --wait exps/260522_density_vit_mae_pretrain/checkpoint_e0099.pth.tar \
#       --watch 260522_density_vit_mae_pretrain \
#       -- bash scripts/train_downstream.sh --density --encoder vit \
#            --encoder-ckpt exps/260522_density_vit_mae_pretrain/checkpoint.pth.tar --gpus 1-6
#
#   # 32a: chain two pretrains (compose with && and a post-gate)
#   bash scripts/00_watch_n_launch.sh \
#       --wait exps/260526_density_vit_mae_40m_xray_pretrain/checkpoint_e0099.pth.tar \
#       --watch 260526_density_vit_mae_40m_xray_pretrain \
#       --expect exps/260526_atomblob_vit_mae_40m_pretrain/checkpoint_e0099.pth.tar \
#       -- bash scripts/pretrain.sh --mode atomblob --gpus 1-5 \
#     && bash scripts/00_watch_n_launch.sh \
#       --expect exps/260526_atomblob_density_vit_mae_40m_pretrain/checkpoint_e0099.pth.tar \
#       -- bash scripts/pretrain.sh --mode atomblob_density --gpus 1-5
#
#   # 37a: gate, then precompute (+ --expect its output) + v2 retrain in one compound command
#   bash scripts/00_watch_n_launch.sh --wait exps/RUN/checkpoint_e0099.pth.tar --watch RUN \
#       --expect dataset/data/xray_crops_aligned_v2/stats.json \
#       -- bash -c 'python dataset/00b_density_preprocess.py --version v2 v3 && \
#                   bash scripts/pretrain.sh --mode atomblob_merged_density --weighted --data v2 --gpus 4-7'
set -u

VOX=${VOX:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG=$VOX/log
SELF=$(basename "$0")          # this script's filename — used to exclude watcher infra from proc_alive
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
die(){ echo "$SELF: $*" >&2; exit 2; }

WAITS=(); WATCH=""; INTERVAL=120; GRACE=30; EXPECTS=(); WATCH_LOG=""; DRYRUN=0
CMD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)     WAITS+=("$2"); shift 2;;
        --watch)    WATCH="$2"; shift 2;;
        --interval) INTERVAL="$2"; shift 2;;
        --grace)    GRACE="$2"; shift 2;;
        --expect)   EXPECTS+=("$2"); shift 2;;
        --log)      WATCH_LOG="$2"; shift 2;;
        --dry-run)  DRYRUN=1; shift;;
        --)         shift; CMD=("$@"); break;;
        -h|--help)  awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)          die "unknown arg: $1  (see --help)";;
    esac
done

[[ ${#CMD[@]} -gt 0 ]] || die 'no COMMAND given (expected "... -- CMD ARGS")'
[[ -n "$WATCH" && ${#WAITS[@]} -eq 0 ]] && die "--watch needs at least one --wait gate (it only guards the pre-gate wait)"
[[ -z "$WATCH_LOG" ]] && WATCH_LOG=$LOG/00_watch_n_launch.log
mkdir -p "$LOG"

log(){ echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

if [[ "$DRYRUN" -eq 1 ]]; then
    echo "would wait for : ${WAITS[*]:-<none>}"
    echo "would watch    : ${WATCH:-<none>}  (interval ${INTERVAL}s)"
    echo "would run      : ${CMD[*]}"
    echo "would expect   : ${EXPECTS[*]:-<none>}"
    exit 0
fi

log "watcher started (PID $$)  log=$WATCH_LOG"

# ── Pre-gate: wait for all gate files; abort if the watched process dies first ─
gate_ready(){ local f; for f in "${WAITS[@]}"; do [[ -f "$f" ]] || return 1; done; return 0; }
proc_alive(){
    # True if a process matching $WATCH exists that is NOT watcher infra. We
    # cannot rely on pgrep alone: the pattern is in this watcher's own argv (and
    # its launching shell's), so those self-match. Filter by /proc cmdline.
    local pid cmd
    while read -r pid; do
        [[ -r /proc/$pid/cmdline ]] || continue
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case "$cmd" in *"$SELF"*) continue;; esac   # skip this watcher (and its launching shell)
        return 0
    done < <(pgrep -f "$WATCH" 2>/dev/null)
    return 1
}
if [[ ${#WAITS[@]} -gt 0 ]]; then
    log "waiting for gate(s): ${WAITS[*]}${WATCH:+  (watching: $WATCH)}"
    while ! gate_ready; do
        if [[ -n "$WATCH" ]] && ! proc_alive; then
            log "ABORT: watched process '$WATCH' gone but gate not satisfied"
            exit 1
        fi
        sleep "$INTERVAL"
    done
    log "gate satisfied"
    log "grace sleep ${GRACE}s (teardown)"; sleep "$GRACE"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
log "launching: ${CMD[*]}"
"${CMD[@]}"
RC=$?
log "command returned (exit $RC)"

# ── Post-gate ─────────────────────────────────────────────────────────────────
if [[ "$RC" -ne 0 ]]; then log "command failed (rc=$RC)"; exit "$RC"; fi
for f in "${EXPECTS[@]}"; do
    if [[ ! -f "$f" ]]; then log "ABORT: expected file missing after command: $f"; exit 1; fi
    log "post-gate ok: $f"
done
log "watcher done"
