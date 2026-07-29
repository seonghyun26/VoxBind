#!/usr/bin/env bash
# Drop into the voxbind:dev container with the project bind-mounted at
# /workspace and CWD set to /workspace/voxbind. Any args after the script
# name are run inside the container instead of an interactive shell.
#
# Examples:
#   bash scripts/tools/99_start_docker.sh                          # interactive bash
#   bash scripts/tools/99_start_docker.sh python -c 'import torch' # one-shot
#   GPUS='"device=0,1,2,3"' bash scripts/tools/99_start_docker.sh  # restrict GPUs
#   IMAGE=voxbind:min bash scripts/tools/99_start_docker.sh        # alt image tag
#   SHM_SIZE=32g bash scripts/tools/99_start_docker.sh             # bigger /dev/shm
#
# Note on GPUS: Docker rejects `--gpus device=0,1,2,3` (parses 1,2,3 as a
# Count). The working form is `--gpus '"device=0,1,2,3"'` — the inner double
# quotes must reach Docker literally. Set GPUS to include them, e.g.
#   GPUS='"device=0,1,2,3"'
# (no eval; "$GPUS" expansion below passes the value through unchanged.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"  # .../VoxBind

IMAGE="${IMAGE:-voxbind:dev}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-16g}"

# Default to bash if the caller passed no command.
if [ "$#" -eq 0 ]; then
    set -- bash
fi

# Allocate a TTY only when stdin is a terminal — keeps the script usable from
# non-interactive contexts (CI, pipelines) without `the input device is not a
# TTY` errors.
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
    --gpus "$GPUS" \
    --shm-size="$SHM_SIZE" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e MPLCONFIGDIR=/tmp/mpl-cache \
    -e PYTHONUNBUFFERED=1 \
    -v "$PROJECT_ROOT":/workspace \
    -w /workspace/voxbind \
    "$IMAGE" "$@"
