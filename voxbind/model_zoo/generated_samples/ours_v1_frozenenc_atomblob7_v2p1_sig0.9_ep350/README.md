# Generated samples — Ours v1 (`voxbind_frozenenc_atomblob7_v2p1_sig0.9`, ep350)

The de-novo molecules behind the gold **"Ours"** row of Table 4 in
`notebook/html/results.html`. Backed up out of `exps/` so they survive checkpoint and
scratch cleanups — the samples are the evidence; re-generating them needs the GPU box and
does not reproduce the same molecules.

## Provenance

| | |
|---|---|
| training run | `exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9` |
| model | frozen C+D+G encoder (ChannelViT MAE, `260701_plinder_v2p1_box_atomblob7_cdg_channelvit_full_pretrain/checkpoint_e0099.pth.tar`) → VoxBind walk-jump generator |
| σ (`smooth_sigma`) | 0.9 |
| epochs | `num_epochs: 350` → final `checkpoint.pth.tar` (2026-07-04) in the source exp dir |
| sampling | `samples/full_eval_ep350`, 4-GPU chunked (`_run_gpu0..3`) |
| set | 79 x-ray-density CrossDocked pockets (`exps/frozenenc_probes/p79_targets.json`) × 100 molecules |
| copied | 2026-09-05, `rsync -a`, all 565 SDFs verified md5-identical to the source |

`cfg.yaml` and `.hydra/` are the training run's, not the sampler's.

## Layout

```
samples/target_XX/
  samples.sdf                     100 generated molecules
  <receptor>_pocket10.pdb         pocket10 crop the numbers were docked against
  <receptor>_pocket10.pdbqt/.pqr  Vina/pdb2pqr receptor prep artefacts
  <ligand>.sdf                    crystal reference ligand (High-Affinity baseline)
  metrics.json
samples/_run_gpu{0..3}/           per-chunk sampling logs
samples/eval_docking_results.json         crop / exh 16  <- the results.html numbers
samples/eval_docking_results_full.json    full receptor / exh 16, 78 pockets
samples/eval_docking_results_full79.json  full receptor / exh 32, 79 pockets (260903 baseline protocol)
samples/eval_docking_results_rerun.json   2026-09-05 independent re-dock of the crop protocol
```

`.vina_tmp_*` scratch (~478 MB) is deliberately not copied.

## Published numbers (`eval_docking_results.json`, crop / exh 16, 79 pockets)

Vina Score **−5.5134** · Min **−7.0898** · Dock **−8.2382** · High-Aff **67.52 %** ·
QED 0.5226 · SA 0.6774 · Diversity 0.7098 · validity 0.9990 · 7,873 valid molecules.

## Pose quality — PoseCheck + PoseBusters, all 79 pockets (2026-09-09)

`target_XX/metrics.json` now carries both, for every molecule and for the pocket's crystal
reference ligand, measured against the pocket10 crop (`pose_scope=crop`, the scope every
published PoseCheck number here uses). PoseBusters had only ever run on 33 of the 79
pockets; `voxbind/scripts/85_fill_pose_eval_4runs.sh` filled the rest and the same files
were re-synced here (md5-identical to `exps/`, 79/79).

PoseCheck clash median **5.0** · strain median **81.9** kcal/mol (1.3.1 definition, not the
paper's — see the strain note in the eval docs) · PoseBusters dock-mode validity
**67.5 %** over 7,873 molecules.

Read the validity against molecule size, not pooled: this run draws bigger molecules than
the baselines (24.9 heavy atoms vs TargetDiff's 22.5), and every pose check gets harder
with size. Per heavy-atom bin, <=15 / 16-20 / 21-25 / 26-30 / >30:

| run | atoms | <=15 | 16-20 | 21-25 | 26-30 | >30 |
|---|---|---|---|---|---|---|
| TargetDiff | 22.5 | 84.9 % | 66.7 % | 57.9 % | 46.8 % | 37.2 % |
| VoxBind σ=0.9 | 24.0 | 84.8 % | 75.3 % | 69.1 % | 61.8 % | 56.2 % |
| **Ours v1** | 24.9 | 83.2 % | 76.3 % | 67.7 % | 61.6 % | 56.1 % |

Pooled, vanilla looks 1.6 points ahead (69.1 % vs 67.5 %); inside a size bin the two are
within noise of each other and both are well clear of TargetDiff. The pooled gap is the
size mix, not pose quality.

## Re-check — both protocols reproduce (2026-09-05 → 09-07)

Independently re-docked from the SDFs above with pinned vina 1.2.2, once per published
protocol. Reports: `redock_vs_published.md`, `redock_full79_vs_published.md`.

| | published | re-run | delta |
|---|---|---|---|
| **crop / exh 16** (`results.html` T4) Score / Min | −5.5134 / −7.0898 | −5.5134 / −7.0898 | **0 / 0** (7,774/7,774 bit-exact) |
| crop Dock, as-scored | −8.2382 | −8.2562 | −0.018 ± 0.042 |
| crop Dock, failed poses dropped | −8.2882 | −8.2881 | **+0.0000 ± 0.0060** |
| **full / exh 32** (`results_drug_design.html`) Score / Min | −6.5839 / −7.6440 | −6.5839 / −7.6440 | **0 / 0** (7,873/7,873 bit-exact) |
| full Dock, as-scored | −8.4886 | −8.4411 | +0.047 ± 0.074 |
| full Dock, failed poses dropped | −8.5186 | −8.5139 | **+0.0047 ± 0.0047** |

(95% CI, paired over pockets.)

Vina's `dock` mode runs with `seed=0` = a **random** seed, so `vina_dock` is not
bit-reproducible by design; `score_only`/`minimize` are deterministic and are the real test
of whether the pipeline (vina build, receptor prep, docking box) is the published one. Both
reproduced exactly, so the published numbers stand.

### Caveat worth carrying into the paper

All of the apparent run-to-run instability is **~0.15 % of molecules where Vina's search
fails and returns a large positive affinity** (up to +232 kcal/mol): 13 published / 8 re-run
under crop, 10 / 12 under full, and they land on *different* molecules each run (2 and 1
molecules shared). Those few values dominate the spread — per-molecule sd falls from
1.88 → 0.25 (crop) and 3.22 → 0.23 (full) once they are excluded.

They also bias the published means **pessimistically**, because a failed pose is a large
positive number averaged in with negative affinities: Dock is −8.238 as-scored vs −8.288
clean (crop) and −8.489 vs −8.519 (full). Dropping non-negative dock affinities is the
usual convention and would both improve the number ~0.03–0.05 kcal/mol and remove the
instability.

## Git

Everything here except this README is gitignored (`*.sdf`/`*.pdb` at the repo root,
`*.json`/`*.log`/`*.yaml`/`*.pdbqt`/`*.pqr` in `model_zoo/.gitignore`). One sample set is
~97k files; it travels via Dropbox (`dropbox_push.sh`), not git.
