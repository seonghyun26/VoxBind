#!/bin/bash
# 10_pretrain_ablation.sh — merged-density / density data-version ablation sweep.
# Preset wrapper over pretrain.sh; collapses the old per-arm scripts 36-39.
#
# Each arm is one pretrain.sh invocation with a sensible default GPU set. Pick an
# arm (or --all to run them sequentially), optionally override --gpus, and pass
# any extra flags through to pretrain.sh (e.g. --dry-run, --epochs, --name, --tags).
#
# Usage:
#   bash scripts/10_pretrain_ablation.sh --list
#   bash scripts/10_pretrain_ablation.sh --arm ARM [--gpus G] [pretrain.sh args...]
#   bash scripts/10_pretrain_ablation.sh --all      [--gpus G] [pretrain.sh args...]
#
# Arms (was):
#   merged_v1   atomblob_merged_density --weighted --data v1   (36)     gpus 4,5,6,7
#   merged_v2   atomblob_merged_density --weighted --data v2   (37)     gpus 4,5,6,7
#   merged_v3   atomblob_merged_density --weighted --data v3   (38)     gpus 4,5,6,7
#   merged_v4   atomblob_merged_density --weighted --data v4   (37_v4)  gpus 4,5,6,7
#   density_v2  atomblob_density (vanilla)         --data v2   (39)     gpus 5
#
# Examples:
#   bash scripts/10_pretrain_ablation.sh --arm merged_v3 --dry-run
#   bash scripts/10_pretrain_ablation.sh --arm merged_v4 --gpus 0,1,2
#   bash scripts/10_pretrain_ablation.sh --all --gpus 4,5,6,7
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRETRAIN=$HERE/pretrain.sh
die(){ echo "10_pretrain_ablation.sh: $*" >&2; exit 1; }

ARMS="merged_v1 merged_v2 merged_v3 merged_v4 density_v2"
# arm → "<pretrain flags>|<default gpus>"
arm_spec(){
    case "$1" in
        merged_v1)  echo "--mode atomblob_merged_density --weighted --data v1|4,5,6,7";;
        merged_v2)  echo "--mode atomblob_merged_density --weighted --data v2|4,5,6,7";;
        merged_v3)  echo "--mode atomblob_merged_density --weighted --data v3|4,5,6,7";;
        merged_v4)  echo "--mode atomblob_merged_density --weighted --data v4|4,5,6,7";;
        density_v2) echo "--mode atomblob_density --data v2|5";;
        *) return 1;;
    esac
}

ARM=""; GPUS=""; ALL=0; PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)     ARM="$2"; shift 2;;
        --gpus)    GPUS="$2"; shift 2;;
        --all)     ALL=1; shift;;
        --list)    for a in $ARMS; do printf '  %-11s %s\n' "$a" "$(arm_spec "$a" | sed 's/|/  [default gpus /;s/$/]/')"; done; exit 0;;
        -h|--help) awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)         PASS+=("$1"); shift;;
    esac
done

run_arm(){
    local a="$1" spec flags defgpus gpus
    spec=$(arm_spec "$a") || die "unknown arm '$a' (see --list)"
    flags=${spec%|*}; defgpus=${spec#*|}
    gpus=${GPUS:-$defgpus}
    echo "[ablation] arm=$a  ->  pretrain.sh $flags --gpus $gpus ${PASS[*]:-}"
    if [[ ${#PASS[@]} -gt 0 ]]; then
        bash "$PRETRAIN" $flags --gpus "$gpus" "${PASS[@]}"
    else
        bash "$PRETRAIN" $flags --gpus "$gpus"
    fi
}

if [[ "$ALL" -eq 1 ]]; then
    for a in $ARMS; do run_arm "$a" || die "arm $a failed"; done
elif [[ -n "$ARM" ]]; then
    run_arm "$ARM"
else
    die "specify --arm ARM, --all, or --list"
fi
