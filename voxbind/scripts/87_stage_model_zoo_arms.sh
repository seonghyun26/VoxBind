#!/usr/bin/env bash
# 87_stage_model_zoo_arms.sh
#   Copy the four svr12 GENERATOR checkpoints out of exps/ into model_zoo/, ready
#   for voxbind/model_zoo/dropbox_push.sh -> dropbox:/박성현/VoxBind/model_zoo.
#
#   WHY: exps/ is git-ignored and backed up nowhere. These four are the only copy
#   of the models behind the sample sets in results/task2-drugdesign, on a shared
#   box whose disk sits at 94%. The SDFs survive in the results bundle; without
#   the weights the arms could never be re-sampled or extended.
#
#   Same layout as every other zoo entry -- cfg.yaml + .hydra/ + the train log +
#   the weights -- plus a MODELCARD.md generated from the staged
#   results/task2-drugdesign/<arm>/metrics.json, so the card's numbers and the
#   bundle's numbers cannot drift apart. Run 86_stage_results_bundle.sh first.
#
#   ~6.1 GB of copies. cp -n: an existing file is never overwritten.
#
#   Usage:  bash voxbind/scripts/87_stage_model_zoo_arms.sh [-n]
set -euo pipefail

ROOT=/home/shpark/prj-denovo/Voxbind
VOX="$ROOT/voxbind"
ZOO="$VOX/model_zoo"
T2="$ROOT/results/task2-drugdesign"

DRY=0
for a in "$@"; do case "$a" in -n|--dry-run) DRY=1 ;; esac; done
run() { if [[ "$DRY" == 1 ]]; then echo "   [dry] $*"; else "$@"; fi; }

# zoo folder | exp dir | results-bundle folder
ARMS=(
  "gen_voxbind_base_ep350|260827_voxbind_base_8gpu|VoxBind-base-ep350"
  "gen_fusion_v4_cdgv2_warm_ep100|260831_fusion_v4_cdgv2_8gpu|Fusion-v4-cdgv2-warm-ep100"
  "gen_fusion_v4_cdgv2_scratch_ep350|260902_fusion_v4_cdgv2_scratch_8gpu|Fusion-v4-cdgv2-scratch-ep350"
  "gen_fusion_v4_cv2_scratch_ep350|260905_fusion_v4_cv2_scratch_8gpu|Fusion-v4-cv2-scratch-ep350"
)

echo "==> staging generator checkpoints into $ZOO  (dry-run=$DRY)"
avail=$(df -BG --output=avail "$ZOO" | tail -1 | tr -dc 0-9)
need=$(du -cBG "$VOX"/exps/{260827_voxbind_base_8gpu,260831_fusion_v4_cdgv2_8gpu,260902_fusion_v4_cdgv2_scratch_8gpu,260905_fusion_v4_cv2_scratch_8gpu}/checkpoint.pth.tar 2>/dev/null | tail -1 | tr -dc 0-9)
echo "    need ~${need} GB, ${avail} GB free"
[[ "$avail" -lt $(( need + 20 )) ]] && { echo "!! not enough headroom, aborting"; exit 1; }

for row in "${ARMS[@]}"; do
  IFS='|' read -r name exp bundle <<<"$row"
  src="$VOX/exps/$exp"
  dst="$ZOO/$name"
  [[ -f "$src/checkpoint.pth.tar" ]] || { echo "!! no checkpoint in $src, skipped"; continue; }
  echo "-- $exp -> model_zoo/$name  ($(du -h "$src/checkpoint.pth.tar" | cut -f1))"
  run mkdir -p "$dst"
  run cp -n --preserve=timestamps "$src/cfg.yaml" "$dst/cfg.yaml"
  [[ -d "$src/.hydra" ]] && run rsync -a "$src/.hydra/" "$dst/.hydra/"
  [[ -f "$src/train_ddp.log" ]] && run cp -n --preserve=timestamps "$src/train_ddp.log" "$dst/train_ddp.log"
  run cp -n --preserve=timestamps "$src/checkpoint.pth.tar" "$dst/checkpoint.pth.tar"
done

if [[ "$DRY" == 0 ]]; then
  echo "-- writing MODELCARD.md from the staged results metrics"
  python3 - "$ZOO" "$T2" "$VOX" <<'PY'
import json, os, sys, time
zoo, t2, vox = sys.argv[1:4]
arms = [
    ("gen_voxbind_base_ep350", "260827_voxbind_base_8gpu", "VoxBind-base-ep350",
     "Vanilla VoxBind, the ICML'24 recipe retrained here",
     "Stock `configs/config_train.yaml`: density-free `model/voxbind` (111.6M), "
     "crossdocked, 64^3 @ 0.25 A, sigma=0.9, lr 1e-5, wd 1e-2, aug, 350 ep, 8-GPU DDP "
     "(bsz 8/rank = 64 effective). Trained by `scripts/70_train_voxbind_base_8gpu.sh`.",
     "The density-free control every fusion arm is measured against -- and the only "
     "arm that covers all 100 test pockets."),
    ("gen_fusion_v4_cdgv2_warm_ep100", "260831_fusion_v4_cdgv2_8gpu", "Fusion-v4-cdgv2-warm-ep100",
     "Token fusion v4 + frozen CDG_v2 encoder, warm-started",
     "`scripts/75_train_fusion_v4_cdgv2_8gpu.sh`, warm-started from "
     "`gen_voxbind_base_ep350`, 100 ep on the 78.5k x-ray subset, bsz 16/rank = 128. "
     "The frozen encoder's PATCH TOKENS feed the denoiser directly "
     "(`models/voxbind.py::_density_token_grid`) -- the representation the affinity "
     "probe was scored on.",
     "First density-conditioned arm. Came out ~0.5 kcal/mol WORSE on Vina dock than the "
     "vanilla baseline over the same 79 pockets while val miou rose, which is what "
     "motivated the two scratch arms: warm-starting confounds density conditioning, the "
     "subset restriction and 100 epochs of drift."),
    ("gen_fusion_v4_cdgv2_scratch_ep350", "260902_fusion_v4_cdgv2_scratch_8gpu", "Fusion-v4-cdgv2-scratch-ep350",
     "Same fusion, trained from scratch",
     "`scripts/80_chain_scratch_v4_full.sh` (WARM_START=\"\"), 350 ep so it matches the "
     "from-scratch vanilla baseline; ~686 s/epoch, ~2.8 days on 8 GPUs.",
     "Removes the warm-start drift term: the density branch is learned jointly with the "
     "denoiser instead of bolted onto a converged model."),
    ("gen_fusion_v4_cv2_scratch_ep350", "260905_fusion_v4_cv2_scratch_8gpu", "Fusion-v4-cv2-scratch-ep350",
     "Coords-only control (C_v2) through the identical fusion",
     "`scripts/81_chain_cv2_scratch_after_sampling.sh`. C_v2 = the same 100M ChannelViT "
     "trunk pretrained on the same PLINDER v2 data but on coords ONLY "
     "(input_mode=atomblob, n_in=11, groups [7,4]) -- no rho, no ||grad rho||. Same "
     "fusion, capacity, subset, schedule and seed as the arm above.",
     "The matched control for the density question: the gap against "
     "`gen_fusion_v4_cdgv2_scratch_ep350` is attributable to the DENSITY CHANNELS rather "
     "than to \"some frozen encoder helps\"."),
]
for name, exp, bundle, headline, recipe, why in arms:
    d = f"{zoo}/{name}"
    if not os.path.isdir(d):
        continue
    m = json.load(open(f"{t2}/{bundle}/metrics.json"))
    e, pb = m["eval"], m["posebusters_260910"]
    ck = f"{d}/checkpoint.pth.tar"
    sz = os.path.getsize(ck) / 2**30
    mt = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(ck)))
    # a fusion arm cannot be loaded without its frozen encoder -- name it, so the
    # zoo entry is self-contained as a *pair* even though the weights are not copied
    dep = ""
    for line in open(f"{d}/cfg.yaml"):
        if "density_pretrained_path:" in line and "null" not in line:
            enc = line.split(":", 1)[1].strip()
            zoo_rel = enc.split("model_zoo/", 1)[-1] if "model_zoo/" in enc else enc
            dep = (f"| frozen encoder | `model_zoo/{zoo_rel}` -- **required to load this "
                   f"checkpoint**; already in the zoo and on Dropbox |\n")
            break
    def row(scope):
        r = pb.get(scope) or {}
        return (f"| {scope} | {r.get('n_pockets','--')} | {r.get('n_molecules','--')} | "
                f"{r.get('vina_score_mean',0):.2f} | {r.get('vina_dock_mean',0):.2f} | "
                f"{r.get('pb_valid_pct',0):.1f} % | {r.get('clash_median','--')} | "
                f"{r.get('strain_median',0):.1f} |")
    open(f"{d}/MODELCARD.md", "w").write(f"""# {name}

{headline}. Generator checkpoint for the **{m['method']}** row of
`results/task2-drugdesign/{bundle}` (samples + per-molecule metrics live there).

| | |
|---|---|
| source run | `voxbind/exps/{exp}` (svr12) |
| weights | `checkpoint.pth.tar`, {sz:.1f} GiB, {mt} |
| sigma | 0.9 |
| samples | `results/task2-drugdesign/{bundle}/samples/` -- {m['n_targets']} pockets x 10 molecules |
{dep}

## Recipe

{recipe}

## Why it exists

{why}

## Generation numbers

Vina in kcal/mol, mean over molecules; PB = PoseBusters dock-mode validity;
clash/strain are PoseCheck medians. `all` = every pocket this arm sampled,
`density79` = the 79 pockets with usable deposited density (the apples-to-apples
scope across arms). Source: `results/task2-drugdesign/{bundle}/metrics.json`.

| scope | pockets | mols | vina score | vina dock | PB valid | clash med | strain med |
|---|---|---|---|---|---|---|---|
{row('all')}
{row('density79')}

QED {e.get('qed_mean',0):.3f} - SA {e.get('sa_mean',0):.3f} - diversity {e.get('diversity',0):.3f} -
high-affinity {100*e.get('high_affinity',0):.1f} % - validity {e.get('validity',0):.3f} -
docked {e.get('vina_n_docked','--')}/{e.get('n_molecules','--')} ({e.get('vina_n_failed','--')} failed).

Our arms draw 10 molecules per pocket against the published baselines' ~100, so
compare per-molecule means, not totals.

## Re-sample from it

```bash
cd voxbind
python sample.py pretrained_path=model_zoo/{name} wjs.split=test \\
    wjs.n_samples_per_pocket=10 wjs.n_targets=79
```

`cfg.yaml` + `.hydra/` are this run's training config; `train_ddp.log` is its log.
Weights and configs are git-ignored here and travel via `dropbox_push.sh`.
""")
    print(f"   {name}/MODELCARD.md")
PY
fi

echo
echo "==> next:  bash voxbind/model_zoo/dropbox_push.sh        # dry-run, then confirms"
du -sh "$ZOO"/gen_* 2>/dev/null || true
