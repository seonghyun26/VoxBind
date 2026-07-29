#!/usr/bin/env bash
# watch_probe_plinder_v2.sh — 260627
# Wait for the v2 (110K) roleblob-DIVERSE PLINDER OTF pretrain to FINISH, then fire the
# frozen-encoder affinity probe on lp_edrscc_v2 — the 110K-vs-17.5K (v2-vs-v1) test.
# Mirrors watch_probe_roleblob_diverse.sh; longer poll cap for the 6.4x-larger corpus.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
cd "$VOX"

EXP=260627_plinder_v2_roleblob_diverse_cdg_channelvit_mask050_pretrain
EXP_DIR="$VOX/exps/$EXP"
COND=roleblob_density_gradmag_channelvit
SPLIT=lp_edrscc_v2
GPU="${GPU:-4}"                       # free once the 4-7 pretrain exits
PY=/home/shpark/.conda/envs/voxbind/bin/python
LOG="$VOX/log/260627_plinder_v2_probe_watch.log"

ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

say "watcher start — waiting for v2 pretrain ($EXP) to exit"

# 1) wait for the v2 training procs to be gone. Match the FULL exp name (the watcher's own
#    cmdline is 'bash …watch_probe_plinder_v2.sh' and does NOT contain it → no self-match).
#    Poll 300s, ~60h cap (110K OTF @ ~100ep can run ~20-30h).
for _ in $(seq 1 720); do
    pgrep -f "$EXP" >/dev/null 2>&1 || break
    sleep 300
done
if pgrep -f "$EXP" >/dev/null 2>&1; then
    say "ERROR: v2 pretrain still running after 60h cap — aborting"; exit 1
fi
say "v2 pretrain process gone"

# 2) final completed epoch from the in-exp train log
E=$(grep -oE ">> epoch: [0-9]+" "$EXP_DIR/train_density.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
[ -z "${E:-}" ] && { say "ERROR: could not parse final epoch from train log"; exit 1; }
say "final completed epoch = $E"
[ "$E" -lt 99 ] && say "WARN: stopped at epoch $E (<99) — probing this checkpoint anyway"

# 3) named checkpoint alias the probe launcher expects (hard-link the rolling ckpt; no extra disk)
EE=$(printf '%04d' "$E")
NAMED="$EXP_DIR/checkpoint_e${EE}.pth.tar"
if [ ! -f "$NAMED" ]; then
    ln "$EXP_DIR/checkpoint.pth.tar" "$NAMED" 2>/dev/null \
        || cp "$EXP_DIR/checkpoint.pth.tar" "$NAMED"
    say "aliased checkpoint.pth.tar -> checkpoint_e${EE}.pth.tar"
fi

# 4) features + 3-seed MLP affinity probe on lp_edrscc_v2 (Kd/Ki canonical; num_workers 0 per probe-perf note)
say "launching affinity probe: cond=$COND split=$SPLIT epoch=$E gpu=$GPU num_workers=0"
bash scripts/04_probe.sh --exp "$EXP" --condition "$COND" --tasks affinity \
    --split "$SPLIT" --epoch "$E" --seeds 3 --gpu "$GPU" --num_workers 0 \
    >> "$LOG" 2>&1
say "probe done — results in dataset/data/pdbbind/results/ (grep $EXP)"
