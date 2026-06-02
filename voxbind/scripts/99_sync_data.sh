#!/bin/bash
# 99_sync_data.sh — move VoxBind dataset payloads between svr7 and a remote box.
#
# Topology / why "pull" is the default:
#   svr7 sits behind NAT/VPN (LAN 192.168.0.7) but is reachable from the internet
#   at 59.29.246.29:22228 (SSH port-forward -> svr7:22). A remote box that can only
#   make OUTBOUND connections (e.g. a NAVER Cloud VM: no inbound, no sudo, no sshd)
#   cannot be pushed to, so it must PULL from svr7.
#
# Modes:
#   pull  (default)  run ON the data-less machine; pulls FROM svr7.
#   push             run ON svr7; pushes TO a remote that accepts inbound SSH.
#
# Remote prerequisites (pull side): rsync + an ssh client. If ssh is missing but
# conda is present (root-free):    conda install -y -c conda-forge openssh
#
# Connectivity notes:
#   * Run inside tmux — password auth is interactive.
#   * IPQoS=throughput works around "kex_exchange_identification: Connection reset
#     by peer" through NAT/cloud-egress paths (the default af21/cs1 DSCP marking on
#     the handshake packets gets dropped). If it still resets, try IPQoS=none.
#   * --partial + the retry loop make the ~46G transfer resumable across drops.
#
# Scope: default is the ~46G "v4" manifest. val_voxels_*/channel_freq_* are
# regenerable caches and are intentionally NOT synced.
#
# Usage:
#   bash voxbind/scripts/99_sync_data.sh                       # pull v4 manifest -> ./dataset/data
#   ITEMS="data_train.pt data_test.pt" bash .../99_sync_data.sh   # custom scope
#   MODE=push REMOTE=user@host PORT=22 REMOTE_DATA=/abs/.../dataset/data bash .../99_sync_data.sh

set -euo pipefail

MODE="${MODE:-pull}"

# --- svr7 = canonical data source (its public SSH forward) ---
SVR7_USER="${SVR7_USER:-shpark}"
SVR7_HOST="${SVR7_HOST:-59.29.246.29}"
SVR7_PORT="${SVR7_PORT:-22228}"
SVR7_DATA="${SVR7_DATA:-/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data}"

# --- local repo data dir (pull destination / push source) ---
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # repo root (script lives in voxbind/scripts/)
LOCAL_DATA="${LOCAL_DATA:-${LOCAL_ROOT}/voxbind/dataset/data}"

# --- what to move (default = ~46G v4 manifest) ---
ITEMS="${ITEMS:-data_train.pt data_test.pt crossdocked_pocket10 xray_crops_aligned_v4}"

SSH_CMD="ssh -o StrictHostKeyChecking=accept-new -o IPQoS=throughput -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=30"
RSYNC_OPTS="-ah --partial --info=progress2"
RETRIES="${RETRIES:-20}"

# run_rsync <port> <src...> <dest> : rsync with a resume-on-failure retry loop
run_rsync() {
    local port="$1"; shift
    local n=0 rc=0
    until rsync ${RSYNC_OPTS} -e "${SSH_CMD} -p ${port}" "$@"; do
        rc=$?
        n=$((n + 1))
        if [ "$n" -ge "$RETRIES" ]; then
            echo "!! giving up after ${RETRIES} retries (last rsync rc=${rc})" >&2
            return 1
        fi
        echo "-- rsync stopped (rc=${rc}); resuming in 10s [retry ${n}/${RETRIES}] --"
        sleep 10
    done
}

case "$MODE" in
    pull)
        echo "==> PULL  ${SVR7_USER}@${SVR7_HOST}:${SVR7_PORT}:${SVR7_DATA}"
        echo "          -> ${LOCAL_DATA}"
        echo "    items: ${ITEMS}"
        mkdir -p "${LOCAL_DATA}"
        srcs=()
        for it in ${ITEMS}; do srcs+=("${SVR7_USER}@${SVR7_HOST}:${SVR7_DATA}/${it}"); done
        run_rsync "${SVR7_PORT}" "${srcs[@]}" "${LOCAL_DATA}/"
        ;;
    push)
        : "${REMOTE:?push mode needs REMOTE=user@host (a box that accepts inbound SSH)}"
        PORT="${PORT:-22}"
        : "${REMOTE_DATA:?push mode needs REMOTE_DATA=/abs/path/.../voxbind/dataset/data}"
        echo "==> PUSH  ${LOCAL_DATA}"
        echo "          -> ${REMOTE}:${REMOTE_DATA} (port ${PORT})"
        echo "    items: ${ITEMS}"
        ${SSH_CMD} -p "${PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DATA}'"
        srcs=()
        for it in ${ITEMS}; do srcs+=("${LOCAL_DATA}/${it}"); done
        run_rsync "${PORT}" "${srcs[@]}" "${REMOTE}:${REMOTE_DATA}/"
        ;;
    *)
        echo "unknown MODE='${MODE}' (use 'pull' or 'push')" >&2
        exit 2
        ;;
esac

echo "==> done."
