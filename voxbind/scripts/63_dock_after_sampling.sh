#!/usr/bin/env bash
# 63_dock_after_sampling.sh — wait for a 62_eval_frozenenc_4gpu.sh sampling run to
# finish, then run the Vina docking eval on its samples.
#
#   SAMPLE_DIR=/home1/irteam/VoxBind/voxbind/exps/samples_scratch_protfirst_ep350 \
#     bash scripts/63_dock_after_sampling.sh
#
# Sampling is GPU-bound and docking is CPU-bound, so chaining them here keeps the
# box busy instead of leaving the GPUs idle waiting for someone to notice.
#
# Docking runs under the isolated voxdock env (vina 1.2.2 + TargetDiff); the driver
# restores the np.int alias in-process for numpy>=1.24.
set -uo pipefail

: "${SAMPLE_DIR:?set SAMPLE_DIR (the eval sample dir containing _run_gpu*/ and target_*/)}"
DRIVER=${DRIVER:-/home1/irteam/VoxBind/voxbind/exps/frozenenc_probes/run_docking_eval.py}
PY=${PY:-/opt/conda/envs/voxdock/bin/python}
POLL=${POLL:-300}
TIMEOUT=${TIMEOUT:-86400}

LOG="$SAMPLE_DIR/dock_chain.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

[ -x "$PY" ]     || { echo "[63_dock] MISSING $PY"; exit 1; }
[ -f "$DRIVER" ] || { echo "[63_dock] MISSING $DRIVER"; exit 1; }

# TargetDiff's VinaDockingTask SHELLS OUT to pdb2pqr30 (and obabel) to protonate the
# receptor. Calling the env's python by absolute path is not enough — without the env's
# bin on PATH those subprocess calls raise FileNotFoundError, which dock_all_modes
# swallows into `None`, so every affinity comes back None in 0.0s and the run still
# "succeeds". Put the env's bin first so the binaries resolve.
export PATH="$(dirname "$PY"):$PATH"
for bin in pdb2pqr30 obabel; do
    command -v "$bin" >/dev/null || { echo "[63_dock] MISSING $bin on PATH"; exit 1; }
done

# Every VoxBind Vina number in the paper is produced with vina 1.2.2 (the voxdock env).
# A different build shifts absolute affinities — 1.2.7 lives in funcbind/.repro-env and is
# easy to pick up by accident — so refuse to run rather than silently mix versions into one
# table. Restore with scripts/rebuild_voxdock_env.sh, or from the data-vol1 archive:
#   zstd -dc .../env-backups/voxdock-env-vina1.2.2.tar.zst | tar -C /opt/conda/envs -xf -
VINA_REQ=${VINA_REQ:-1.2.2}
vina_ver=$("$PY" -c 'import importlib.metadata as m; print(m.version("vina"))' 2>/dev/null)
if [ "$vina_ver" != "$VINA_REQ" ]; then
    echo "[63_dock] WRONG VINA: $PY has vina=${vina_ver:-none}, need $VINA_REQ"
    exit 1
fi
mkdir -p "$SAMPLE_DIR"

say "waiting for sampling chunks under $SAMPLE_DIR"
waited=0
while :; do
    n_done=$(ls "$SAMPLE_DIR"/_run_gpu*/exit_code 2>/dev/null | wc -l)
    n_chunk=$(ls -d "$SAMPLE_DIR"/_run_gpu* 2>/dev/null | wc -l)
    if (( n_chunk > 0 && n_done >= n_chunk )); then
        say "all $n_chunk chunks finished"
        break
    fi
    if (( waited >= TIMEOUT )); then
        say "TIMEOUT after ${waited}s with $n_done/$n_chunk chunks done - not docking"
        exit 1
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done

for f in "$SAMPLE_DIR"/_run_gpu*/exit_code; do
    rc=$(cat "$f")
    [ "$rc" = "0" ] || say "WARNING: $(dirname "$f" | xargs basename) exited $rc (docking what it produced)"
done

n_t=$(ls -d "$SAMPLE_DIR"/target_* 2>/dev/null | wc -l)
say "docking $n_t target dirs (this is CPU-bound; GPUs are free for other work)"
# Extra driver flags. With the parallel driver this is where the worker count goes:
# the box is capped at 32 cores by its cgroup quota (nproc reports 128 and lies), so
# leaving headroom here is what keeps a co-running GPU job's dataloader fed.
"$PY" "$DRIVER" "$SAMPLE_DIR" ${ARGS:-} >>"$LOG" 2>&1
rc=$?
say "docking eval exit=$rc -> $SAMPLE_DIR/eval_docking_results.json"
exit $rc
