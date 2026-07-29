#!/usr/bin/env bash
# Hydra-first encoder pretraining launcher.
#
# Recommended:
#   bash scripts/03_pretrain.sh \
#     --experiment cha_gradmag \
#     --gpus 0-3 \
#     -- num_epochs=200 seed=43
#
# Direct config:
#   bash scripts/03_pretrain.sh \
#     --config-name config_train_density_vit_mae \
#     --name density_smoke \
#     --gpus 0 \
#     -- debug=true num_epochs=1
#
# GPU selection may be omitted when VOXBIND_GPUS or CUDA_VISIBLE_DEVICES is
# inherited from scripts/99_chain.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

VOXBIND_CALLER="03_pretrain.sh"
VOX="$(voxbind_repo_root)"
PY="${VOXBIND_PY:-/home/shpark/.conda/envs/voxbind/bin}"
CONFIG_NAME="pretrain"
EXPERIMENT=""
GPUS="${VOXBIND_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
NAME=""
RUN_LOG=""
DRY_RUN=0
HYDRA_ARGS=()

# Preserve the old generated-mode interface without keeping its implementation
# in the stable launcher.
for arg in "$@"; do
    if [[ "$arg" == "--mode" ]]; then
        printf '%s\n' \
            "[deprecated] --mode uses scripts/archive/launchers/03_pretrain_legacy.sh;" \
            "             prefer --experiment with a Hydra preset." >&2
        exec bash "$SCRIPT_DIR/archive/launchers/03_pretrain_legacy.sh" "$@"
    fi
done

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --experiment NAME   Compose configs/experiment/NAME.yaml into the base
                      pretrain config.
  --config-name NAME  Hydra config name (default: pretrain).
  --name NAME         Override exp_name and the default log filename.
  --gpus SPEC         Comma list or range, e.g. 0,1 or 4-7. Inherits the GPU
                      environment selected by 99_chain.sh when omitted.
  --log FILE          Append stdout/stderr to FILE (default: log/<exp_name>.log).
  --dry-run           Print the resolved command without launching.
  -- OVERRIDES...     Hydra overrides, e.g. num_epochs=200 model.depth=24.

Legacy:
  Commands containing --mode are forwarded to the archived compatibility
  launcher. New variants should be Hydra experiment presets.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)  EXPERIMENT="${2:?--experiment requires a value}"; shift 2;;
        --config-name|--config)
                       CONFIG_NAME="${2:?--config-name requires a value}"; shift 2;;
        --name)        NAME="${2:?--name requires a value}"; shift 2;;
        --gpus)        GPUS="${2:?--gpus requires a value}"; shift 2;;
        --log)         RUN_LOG="${2:?--log requires a value}"; shift 2;;
        --dry-run)     DRY_RUN=1; shift;;
        -h|--help)     usage; exit 0;;
        --)            shift; HYDRA_ARGS=("$@"); break;;
        *)             voxbind_die "unknown argument '$1' (use --help)";;
    esac
done

CONFIG_FILE="$VOX/configs/$CONFIG_NAME.yaml"
[[ -f "$CONFIG_FILE" ]] || voxbind_die "Hydra config not found: $CONFIG_FILE"
if [[ -n "$EXPERIMENT" ]]; then
    EXPERIMENT_FILE="$VOX/configs/experiment/$EXPERIMENT.yaml"
    [[ -f "$EXPERIMENT_FILE" ]] ||
        voxbind_die "experiment preset not found: $EXPERIMENT_FILE"
fi
[[ -n "$GPUS" ]] || voxbind_die "--gpus is required unless inherited from scripts/99_chain.sh"

CUDA_LIST="$(voxbind_expand_gpus "$GPUS")" ||
    voxbind_die "invalid --gpus specification '$GPUS'"
NPROC="$(voxbind_gpu_count "$CUDA_LIST")"

if [[ -z "$NAME" && -n "$EXPERIMENT" ]]; then
    NAME="$(awk '$1 == "exp_name:" && $2 != "???" {print $2; exit}' "$EXPERIMENT_FILE")"
fi
if [[ -z "$NAME" ]]; then
    NAME="$(awk '$1 == "exp_name:" && $2 != "???" {print $2; exit}' "$CONFIG_FILE")"
fi
[[ -n "$NAME" ]] ||
    voxbind_die "could not derive exp_name from the config; pass --name"
[[ -n "$RUN_LOG" ]] || RUN_LOG="$VOX/log/$NAME.log"

CMD=(
    "$PY/torchrun"
    --standalone
    --nproc_per_node="$NPROC"
    train_density.py
    --config-name="$CONFIG_NAME"
)
[[ -z "$EXPERIMENT" ]] || CMD+=("+experiment=$EXPERIMENT")
CMD+=("exp_name=$NAME")
(( ${#HYDRA_ARGS[@]} == 0 )) || CMD+=("${HYDRA_ARGS[@]}")

voxbind_log "pretrain config=$CONFIG_NAME experiment=${EXPERIMENT:-none} run=$NAME"
voxbind_log "GPUs=$CUDA_LIST nproc=$NPROC log=$RUN_LOG"
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q \\\n' "$CUDA_LIST"
    voxbind_print_command "${CMD[@]}"
    exit 0
fi

mkdir -p "$(dirname "$RUN_LOG")"
cd "$VOX" || voxbind_die "cannot cd to $VOX"
CUDA_VISIBLE_DEVICES="$CUDA_LIST" "${CMD[@]}" >> "$RUN_LOG" 2>&1
RC=$?
voxbind_log "$NAME finished with exit code $RC"
exit "$RC"
