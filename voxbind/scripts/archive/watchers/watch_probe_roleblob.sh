#!/usr/bin/env bash
# watch_probe_roleblob.sh — 260624
# Wait for the resumed roleblob ChannelViT PLINDER pretrain to finish (epoch 99),
# then fire the frozen-encoder affinity probe on lp_edrscc_v2 and summarise it
# against the matched coords-ChannelViT (C) baseline.
#
#   roleblob C+D+G (4ch: lig-role, poc-role, density, gradmag)  vs  coords C
#
# The pretrain runs ckpt_every=50, so no checkpoint_e0099.pth.tar is written on a
# resume that ends at e99 — we hard-link the rolling checkpoint.pth.tar to the
# named file the probe launcher expects.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
cd "$VOX"

EXP=260623_plinder_otf_roleblob_cdg_channelvit_mask050_pretrain
EXP_DIR="$VOX/exps/$EXP"
CFGNAME=config_train_roleblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
COND=roleblob_density_gradmag_channelvit
SPLIT=lp_edrscc_v2
GPU=5
PY=/home/shpark/.conda/envs/voxbind/bin/python
LOG="$VOX/log/260624_roleblob_probe_watch.log"

ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

say "watcher start — waiting for roleblob pretrain ($CFGNAME) to exit"

# 1) wait for ALL training procs of this config to be gone (poll 120s, ~4h cap)
for _ in $(seq 1 120); do
    pgrep -f "$CFGNAME" >/dev/null 2>&1 || break
    sleep 120
done
if pgrep -f "$CFGNAME" >/dev/null 2>&1; then
    say "ERROR: roleblob still running after cap — aborting"; exit 1
fi
say "roleblob pretrain process gone"

# 2) final completed epoch from the in-exp train log
E=$(grep -oE ">> epoch: [0-9]+" "$EXP_DIR/train_density.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
[ -z "${E:-}" ] && { say "ERROR: could not parse final epoch from train log"; exit 1; }
say "final completed epoch = $E"
[ "$E" -lt 99 ] && say "WARN: stopped at epoch $E (<99) — probing this checkpoint anyway"

# 3) named checkpoint alias for the probe launcher (hard-link, no extra disk)
EE=$(printf '%04d' "$E")
NAMED="$EXP_DIR/checkpoint_e${EE}.pth.tar"
if [ ! -f "$NAMED" ]; then
    ln "$EXP_DIR/checkpoint.pth.tar" "$NAMED" 2>/dev/null \
        || cp "$EXP_DIR/checkpoint.pth.tar" "$NAMED"
    say "aliased checkpoint.pth.tar -> checkpoint_e${EE}.pth.tar"
fi

# 4) features + 3-seed MLP affinity probe on lp_edrscc_v2 (Kd/Ki-only canonical)
say "launching affinity probe: cond=$COND split=$SPLIT epoch=$E gpu=$GPU num_workers=0"
bash scripts/04_probe.sh --exp "$EXP" --condition "$COND" --tasks affinity \
    --split "$SPLIT" --epoch "$E" --seeds 3 --gpu "$GPU" --num_workers 0 \
    >> "$LOG" 2>&1
say "probe exit code = $?"

# 5) summarise roleblob C+D+G vs coords-ChannelViT (C) baseline
say "=== RESULT SUMMARY (roleblob C+D+G  vs  coords C) ==="
"$PY" - >> "$LOG" 2>&1 <<'PY'
import csv, glob, os, statistics as st
res = "dataset/data/pdbbind/results"
def summ(path):
    rows = [r for r in csv.DictReader(open(path))
            if r.get("seed", "").strip() and r["seed"] != "seed"]
    if not rows:
        return None
    sp = [float(r["test_spearman"]) for r in rows]
    pe = [float(r["test_pearson"]) for r in rows]
    return len(sp), rows[0]["n_test"], st.mean(sp), st.pstdev(sp), st.mean(pe)
def show(tag, path):
    s = summ(path)
    if s:
        print(f"  {tag:28s} Spearman={s[2]:.3f}±{s[3]:.3f}  Pearson={s[4]:.3f}  "
              f"(seeds={s[0]}, n_test={s[1]})  [{os.path.basename(path)}]")
    else:
        print(f"  {tag:28s} (no data rows yet) [{os.path.basename(path)}]")
cands = sorted(glob.glob(f"{res}/probe_results_e*_v5_{('lp_edrscc_v2')}split*roleblob*.csv"))
for f in cands:
    show("roleblob C+D+G (CVit)", f)
if not cands:
    print("  roleblob C+D+G (CVit)       NO RESULT CSV FOUND — check probe log above")
show("coords C (CVit) baseline",
     f"{res}/probe_results_e99_v5_lp_edrscc_v2split_coords_channelvit_mask050.csv")
PY
say "watcher done"
