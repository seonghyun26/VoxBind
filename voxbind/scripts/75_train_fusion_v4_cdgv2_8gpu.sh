#!/usr/bin/env bash
# 75_train_fusion_v4_cdgv2_8gpu.sh
#   fusion='v4' (token fusion) with the FROZEN CDG_v2 encoder, on all 8 GPUs of svr12.
#
#   This is scripts/67_train_fusion_champion_reference_4gpu.sh ported to THIS box
#   (different root, conda env, GPU count) with FUSION=v4 and the CDG_v2 encoder:
#
#     x     = ligand_encoder(y) + pocket_encoder(pocket)
#     tok   = token_trunk(spatial_norm(CDG_v2.forward_features(enc_in)))   # g_p³ patch grid
#     x     = x + token_proj(cat([x, broadcast(tok), intra_patch_offsets]))  # zero-init
#     enc_in= [ ligand(7) | pocket(4) | rho | ||grad rho|| ]                 # CDG_v2's 13ch
#
#   v4 vs the default fusion: the frozen encoder's PATCH TOKENS are consumed directly
#   (models/voxbind.py `_density_token_grid`), skipping `_pool_groups` (which mean-pools
#   the channel groups, diluting the density/gradmag tokens) and the frozen MAE
#   `decoder_proj`. That token representation is the one the affinity probe was scored
#   on, so what the generator sees is what was measured.
#
#   ENCODER — CDG_v2 = copy of atombias_100m_v2_e25 (run 260806_cdg_100m_v2_ep100, e25),
#   the paper's headline CDG v2 row: test rho 0.653 / r 0.666 / RMSE 1.355, best-on-record.
#   Its geometry (dim 640 / depth 18 / heads 10, groups [7,4,2]) is hardcoded from the
#   folder's own cfg.yaml — model_zoo entries are NOT interchangeable at fixed dims.
#
#   BATCH — bsz is PER-RANK under train_ddp.py, so 16 x 8 = effective 128, identical to
#   script 67's 32 x 4 on the 4-GPU box. NOT 32: the v4 GPU smoke measured peak 87.4 GiB
#   of 95.0 GiB at bsz=32, i.e. 92% VRAM, which will OOM on fragmentation over a multi-day
#   run. At 16 the same effective batch fits with headroom.
#
#   WARM START — exps/260827_voxbind_base_8gpu (epoch 349), this box's vanilla density-free
#   sig0.9 run; the local equivalent of the other box's exp_sig0.9_v2. token_proj is
#   zero-init, so step 0 is functionally identical to that checkpoint.
#
#   SUBSET_N is computed from the v5 crops' OWN train_available.npy rather than hardcoded
#   to the other box's 78,512: the availability mask depends on the alignment gates, which
#   are recomputed when the crops are rebuilt here. 100 rows are held out as val.
#
#   wjs.n_targets=0 disables mid-training WJS sampling: it runs on rank 0 only while the
#   other 7 ranks sit at the epoch barrier, and blows past the 2h NCCL watchdog.
#
#   PYTHONPATH is set because this env's editable install still maps `voxbind` to the
#   deleted /home/shpark/prj-ligand/Voxbind checkout.
#
#   Env knobs: EXP_NAME, NUM_EPOCHS, BSZ (per-rank), SIGMA, COND_DROPOUT, CROPS_DIR,
#              SUBSET_N, RESUME, MASTER_PORT, ENCODER, FUSION, WANDB.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
PY=/home/shpark/miniforge3/envs/voxbind/bin
cd "$ROOT/voxbind" || exit 1

ENCODER="${ENCODER:-$ROOT/voxbind/model_zoo/CDG_v2/checkpoint_e0025.pth.tar}"
# Geometry is READ FROM THE ENCODER FOLDER'S OWN cfg.yaml, never hardcoded: model_zoo
# entries are not interchangeable at fixed dims, and the channel layout differs by family
#   CDG_v2  n_in=13 groups [7,4,2]  (coords + density + gradmag)
#   CD_v2   n_in=12 groups [7,4,1]
#   C_v2    n_in=11 groups [7,4]    (coords only — the matched no-density control)
# Passing a different ENCODER without its matching VIT_* values loads a state_dict into the
# wrong shapes; deriving them here makes that impossible.
_ENC_CFG="$(dirname "$ENCODER")/cfg.yaml"
[ -f "$_ENC_CFG" ] || { echo "[75_v4] MISSING encoder cfg $_ENC_CFG"; exit 1; }
eval "$("$PY/python" - "$_ENC_CFG" <<'PY'
import sys, yaml
m = (yaml.safe_load(open(sys.argv[1])) or {}); m = m.get("model", m)
g = [int(x) for x in (m.get("channel_groups") or [7, 4, 2])]
print(f'VIT_PATCH={int(m.get("patch_size",8))}')
print(f'VIT_DIM={int(m["dim"])}')
print(f'VIT_DEPTH={int(m["depth"])}')
print(f'VIT_HEADS={int(m["heads"])}')
print(f'VIT_MLP_RATIO={int(m.get("mlp_ratio",4))}')
print(f'VIT_DROPOUT={float(m.get("dropout",0.1))}')
print(f'VIT_NCH={int(m.get("n_in_channels",13))}')
print("VIT_GROUPS='[%s]'" % ",".join(str(x) for x in g))
PY
)"
[ -n "${VIT_DIM:-}" ] || { echo "[75_v4] could not read geometry from $_ENC_CFG"; exit 1; }

# NO_DENSITY=1 → the CONTROL for "does v4 fusion help?". The v4 run does not freeze the
# denoiser — only the encoder is frozen, so its 111.57M denoiser params keep training and
# any epoch-0-vs-epoch-N gain mixes the density branch with plain extra finetuning on the
# x-ray subset. This arm removes ONLY the density branch and holds everything else
# bit-identical (same subset rows via subset_xray_only + subset_n, same bsz/epochs/lr/seed,
# same warm start), so the difference between the two curves is attributable to the
# conditioning. Run it at matched epochs and compare val miou, NOT loss (see the
# reduction="sum" batch-size artifact).
NO_DENSITY="${NO_DENSITY:-}"
FUSION="${FUSION:-v4}"
if [ -n "$NO_DENSITY" ]; then
    EXP_NAME="${EXP_NAME:-260831_fusion_v4_cdgv2_8gpu_nodensity}"
    MASTER_PORT="${MASTER_PORT:-29542}"   # so it can run beside the density arm on 4+4
fi
EXP_NAME="${EXP_NAME:-260831_fusion_v4_cdgv2_8gpu}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BSZ="${BSZ:-16}"
SIGMA="${SIGMA:-0.9}"
COND_DROPOUT="${COND_DROPOUT:-0}"
WARM_START="${WARM_START-$ROOT/voxbind/exps/260827_voxbind_base_8gpu/checkpoint.pth.tar}"
CROPS_DIR="${CROPS_DIR:-$ROOT/voxbind/dataset/data/pretrain/xray_crops_aligned_v5}"
SUBSET_VAL_N="${SUBSET_VAL_N:-100}"
RESUME="${RESUME:-}"
MASTER_PORT="${MASTER_PORT:-29541}"
WANDB="${WANDB:-true}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

mkdir -p logs
LOG="logs/${EXP_NAME}.log"

[ -n "$NO_DENSITY" ] || [ -f "$ENCODER" ] || { echo "[75_v4] MISSING encoder $ENCODER"; exit 1; }
[ -z "$WARM_START" ] || [ -f "$WARM_START" ] || {
  echo "[75_v4] MISSING warm-start checkpoint $WARM_START"; exit 1; }
[ -d "$CROPS_DIR/train" ] || { echo "[75_v4] MISSING v5 crops at $CROPS_DIR/train"; exit 1; }

# Availability count comes from the crops actually on disk, minus the val holdout.
if [ -z "${SUBSET_N:-}" ]; then
  SUBSET_N=$("$PY/python" - "$CROPS_DIR" "$SUBSET_VAL_N" <<'PY'
import sys, numpy as np
n = int(np.load(f"{sys.argv[1]}/train_available.npy").sum())
print(max(n - int(sys.argv[2]), 1))
PY
)
fi
[ -n "$SUBSET_N" ] || { echo "[75_v4] could not derive SUBSET_N"; exit 1; }

RESUME_ARG=()
[ -n "$RESUME" ] && RESUME_ARG=("resume=$RESUME")

echo "[75_v4] exp=$EXP_NAME fusion=$FUSION epochs=$NUM_EPOCHS sigma=$SIGMA cond_dropout=$COND_DROPOUT"
echo "[75_v4] encoder=$ENCODER (frozen, dim=$VIT_DIM depth=$VIT_DEPTH heads=$VIT_HEADS)"
echo "[75_v4] student=${WARM_START:-<scratch>}"
echo "[75_v4] gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC bsz=$BSZ (effective $((BSZ * NPROC)))"
echo "[75_v4] crops=$CROPS_DIR subset=$SUBSET_N+$SUBSET_VAL_N"
echo "[75_v4] log: $ROOT/voxbind/$LOG"

# DRY_RUN=1 resolves the config and exits instead of launching. Use it after ANY edit to
# the override list below: a key that exists in config_train.yaml but not in this run's
# --config-name is rejected by Hydra ("not in struct") and kills all 8 ranks ~7s in, which
# under the supervisor becomes a silent 20-attempt retry loop. Validating a hand-retyped
# SUBSET of the overrides does not catch it — only the real list does.
if [ -n "${DRY_RUN:-}" ]; then
  set -- "$PY/python" train_ddp.py --cfg job
else
  set -- "$PY/torchrun" --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_ddp.py
fi

if [ -n "$NO_DENSITY" ]; then
  # Control arm: density branch off. dset.* is UNCHANGED on purpose — subset_xray_only +
  # subset_n select rows from the crops' availability mask, so both arms see exactly the
  # same molecules; the loader still reads the maps, the model just never consumes them.
  DENSITY_ARGS=(
    model.with_density=false
    wandb_tags="[voxbind,fusion,v4,control,nodensity,8gpu,ddp,sigma${SIGMA}]"
  )
else
  DENSITY_ARGS=(
    model.with_density=true model.density_encoder_type=vit model.density_freeze=true
    model.density_pretrained_path="$ENCODER"
    model.density_vit.patch="$VIT_PATCH" model.density_vit.dim="$VIT_DIM"
    model.density_vit.depth="$VIT_DEPTH" model.density_vit.heads="$VIT_HEADS"
    model.density_vit.mlp_ratio="$VIT_MLP_RATIO" model.density_vit.dropout="$VIT_DROPOUT"
    model.density_vit.n_in_channels="$VIT_NCH"
    model.density_vit.patch_embed_mode=channel_group
    model.density_vit.channel_groups="$VIT_GROUPS"
    model.density_encoder_sees_ligand=true
    model.density_mask_ligand=false
    model.fusion="$FUSION"
    model.density_encoder_amp=true
    ++model.density_cond_dropout="$COND_DROPOUT"
    wandb_tags="[voxbind,fusion,v4,token_fusion,cdg_v2,8gpu,ddp,sigma${SIGMA}]"
  )
fi

exec "$@" \
  --config-name config_train_voxbind_fusion_champion_reference \
  wandb="$WANDB" \
  num_workers=12 prefetch_factor=8 \
  exp_name="$EXP_NAME" \
  num_epochs="$NUM_EPOCHS" bsz="$BSZ" accum_steps=1 lr=1e-5 wd=1e-2 smooth_sigma="$SIGMA" \
  ++ddp_static_graph=false \
  dset.crops_dir="$CROPS_DIR" \
  dset.normalize=false dset.pocket_radius=-1 dset.ligand_radius=0.5 \
  dset.use_xray=true dset.subset_xray_only=true \
  dset.subset_n="$SUBSET_N" dset.subset_val_n="$SUBSET_VAL_N" dset.cache_size=32 \
  "${DENSITY_ARGS[@]}" \
  wjs.n_targets=0 \
  pretrained_path="${WARM_START:-null}" \
  "${RESUME_ARG[@]}" \
  >> "$LOG" 2>&1
