#!/usr/bin/env bash
# Stable pretrain -> frozen-encoder probe workflow.
#
# Example:
#   bash scripts/workflows/pretrain_then_probe.sh \
#     --experiment cha_gradmag \
#     --condition atomblob_density_gradmag \
#     --gpus 0-3 \
#     -- num_epochs=100 seed=42
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPTS_DIR/lib/common.sh"

VOXBIND_CALLER="pretrain_then_probe.sh"
VOX="$(voxbind_repo_root)"
EXPERIMENT=""
NAME=""
CONDITION=""
GPUS="${VOXBIND_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
TASKS="affinity"
SPLIT="lp_edrscc"
EPOCH=99
PROBE_GPU=""
TAG=""
DRY_RUN=0
HYDRA_ARGS=()

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Required:
  --experiment NAME   configs/experiment/NAME.yaml
  --condition NAME    Encoder input condition passed to 04_probe.sh

Options:
  --name NAME         Override Hydra exp_name and probe this directory.
  --gpus SPEC         Training GPUs; inherits VOXBIND_GPUS/CUDA_VISIBLE_DEVICES.
  --tasks LIST        Probe tasks (default: affinity).
  --split NAME        Probe split (default: lp_edrscc).
  --epoch N           Encoder checkpoint epoch (default: 99).
  --probe-gpu N       Physical GPU for probing (default: first training GPU).
  --tag TAG           Probe result tag (default: run name).
  --dry-run           Print both commands without running.
  -- OVERRIDES...     Hydra overrides passed to 03_pretrain.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="${2:?--experiment requires a value}"; shift 2;;
        --name)       NAME="${2:?--name requires a value}"; shift 2;;
        --condition)  CONDITION="${2:?--condition requires a value}"; shift 2;;
        --gpus)       GPUS="${2:?--gpus requires a value}"; shift 2;;
        --tasks)      TASKS="${2:?--tasks requires a value}"; shift 2;;
        --split)      SPLIT="${2:?--split requires a value}"; shift 2;;
        --epoch)      EPOCH="${2:?--epoch requires a value}"; shift 2;;
        --probe-gpu)  PROBE_GPU="${2:?--probe-gpu requires a value}"; shift 2;;
        --tag)        TAG="${2:?--tag requires a value}"; shift 2;;
        --dry-run)    DRY_RUN=1; shift;;
        -h|--help)    usage; exit 0;;
        --)           shift; HYDRA_ARGS=("$@"); break;;
        *)            voxbind_die "unknown argument '$1' (use --help)";;
    esac
done

[[ -n "$EXPERIMENT" ]] || voxbind_die "--experiment is required"
[[ -n "$CONDITION" ]] || voxbind_die "--condition is required"
[[ -f "$VOX/configs/experiment/$EXPERIMENT.yaml" ]] ||
    voxbind_die "unknown experiment preset: configs/experiment/$EXPERIMENT.yaml"
[[ -n "$GPUS" ]] || voxbind_die "--gpus is required unless inherited from 99_chain.sh"

EXPANDED_GPUS="$(voxbind_expand_gpus "$GPUS")" ||
    voxbind_die "invalid --gpus specification '$GPUS'"
if [[ -z "$PROBE_GPU" ]]; then
    PROBE_GPU="${EXPANDED_GPUS%%,*}"
fi
if [[ -z "$NAME" ]]; then
    NAME="$(awk '$1 == "exp_name:" {print $2; exit}' \
        "$VOX/configs/experiment/$EXPERIMENT.yaml")"
fi
[[ -n "$NAME" ]] || voxbind_die "could not derive exp_name; pass --name explicitly"
[[ -n "$TAG" ]] || TAG="$NAME"

PRETRAIN_CMD=(
    bash "$SCRIPTS_DIR/03_pretrain.sh"
    --experiment "$EXPERIMENT"
    --name "$NAME"
    --gpus "$EXPANDED_GPUS"
)
PROBE_CMD=(
    bash "$SCRIPTS_DIR/04_probe.sh"
    --exp "$NAME"
    --condition "$CONDITION"
    --tasks "$TASKS"
    --split "$SPLIT"
    --epoch "$EPOCH"
    --gpu "$PROBE_GPU"
    --tag "$TAG"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
    PRETRAIN_CMD+=(--dry-run)
    PROBE_CMD+=(--dry-run)
fi
(( ${#HYDRA_ARGS[@]} == 0 )) || PRETRAIN_CMD+=(-- "${HYDRA_ARGS[@]}")

voxbind_log "workflow: pretrain experiment=$EXPERIMENT run=$NAME"
"${PRETRAIN_CMD[@]}" || voxbind_die "pretraining failed; probe was not started"
voxbind_log "workflow: probe run=$NAME tasks=$TASKS split=$SPLIT"
"${PROBE_CMD[@]}"
