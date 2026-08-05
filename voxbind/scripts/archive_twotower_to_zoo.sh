#!/usr/bin/env bash
# On two-tower completion: copy BOTH encoder checkpoints out of exps/ into model_zoo/ (so they
# survive checkpoint cleanups, per the model_zoo convention) and push to the SPML Dropbox.
# Idempotent: skips a tower that has no final checkpoint yet; rclone copy is incremental.
#   model_zoo/<name>/{cfg.yaml, .hydra/, train_density.log, checkpoint_e0049.pth.tar}
# usage: archive_twotower_to_zoo.sh   (edit the map below if exp names change)
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
export PATH="$HOME/.local/bin:$PATH"
CK=checkpoint_e0049.pth.tar

# source exp  ->  model_zoo folder name
declare -A ZOO=(
  [260806_tt_pocket_protein_vdw_mc]=twotower_pocket_protein_vdw_mc
  [260806_tt_ligdens_protein_vdw_mc]=twotower_ligdens_protein_vdw_mc
)

archived=0
for exp in "${!ZOO[@]}"; do
  src="exps/$exp"; dst="model_zoo/${ZOO[$exp]}"
  if [ ! -f "$src/$CK" ]; then
    echo "SKIP $exp — no $CK yet (not finished)"; continue
  fi
  mkdir -p "$dst"
  cp -f "$src/cfg.yaml" "$src/$CK" "$dst/"
  cp -f "$src/train_density.log" "$dst/" 2>/dev/null || true
  cp -rf "$src/.hydra" "$dst/" 2>/dev/null || true
  echo "archived $exp -> $dst  ($(du -h "$dst/$CK" | cut -f1))"
  archived=$((archived+1))
done

if [ "$archived" -eq 0 ]; then
  echo "nothing archived (no finished checkpoints) — skipping Dropbox push."; exit 0
fi

echo ">> pushing model_zoo/ to Dropbox (incremental) ..."
bash model_zoo/dropbox_push.sh -y
echo ">> done. (add a metrics row to model_zoo/README.md after the probe result lands.)"
