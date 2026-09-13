# VoxBind — results bundle

Per-method evaluation results for the three VoxBind tasks, plus the rendered
report pages and the Docker/conda environment to run them.

This folder is **git-ignored** and backed up to Dropbox instead (same mechanism
as `voxbind/model_zoo/`). Only `dropbox_push.sh`, `dropbox_pull.sh`,
`dropbox_pull_baselines.sh` and this `README.md` travel via git so a fresh
checkout can bootstrap the pull.

## Layout

```
results/
├── task1-affinity/     <method>/  representations/  metrics.json   (+ SOURCE.txt)
├── task2-drugdesign/   <method>/  samples/  eval/  metrics.json   (+ SOURCE.txt)
│                       EVAL_STATUS.md · EVAL_STATUS.json  = the coverage matrix
├── task3-mcp/          <method>/  samples/          metrics.json
│                       _shared/  = cross-method analysis artifacts (not a method)
├── reports/            results.html · results_drug_design.html · results_mcp.html
├── docker/             Dockerfile · env.yaml · env.minimal.yaml
├── dropbox_push.sh · dropbox_pull.sh · dropbox_pull_baselines.sh · README.md
                                                          (all git-tracked)
```

Each **method** folder holds a `metrics.json` (always) plus its artifacts:
`representations/` (cached encoder features, affinity) or `samples/` (generated
molecules, generation). When the raw artifact isn't on this machine, a
`SOURCE.txt` records where it lives instead.

## task1-affinity  (21 methods)

`metrics.json` = headline LP-PDBBind `lp_edrscc_v2` test metrics — Pearson /
Spearman / RMSE (mean ± std, 3 seeds). Source: `build_appendixB_bar.py::METHODS`
(and `base/dta/result/*` for DeepDTA/MolTrans). The full multi-split tables
(+CL1/CL2/CL3, novelty, CASF) are in `reports/results.html`.

Methods: DeepDTA, MolTrans, HonestAffinity, Nesso-1, HBGSA, EGNN, EGNN-TargetDiff,
GET, CheapNet, BindNet, AEV-PLIG, DSMBind, ProFSA, GeoSSL, IPNet-frozen,
IPNet-scratch, C, C-D-G, C-D-G-corr, CDG-v2 (= Ours), CDG-v3.

`representations/` present for: CDG-v2, C-D-G, C-D-G-corr (voxbind features),
GeoSSL (schnet), IPNet-frozen (IPDiff feats), DeepDTA/MolTrans (DeepPurpose
logits). Others carry a `SOURCE.txt` (re-extract from `base/<method>/`).

## task2-drugdesign  (CrossDocked)

`metrics.json` = the headline row as the reports read it: PoseCheck aggregate
(strain/clash/heavy-atoms) and, for the voxel-diffusion models, Vina Dock over the
79 density pockets. The full per-evaluation numbers live in `<method>/eval/` (below);
`metrics.json` is left exactly as it was so nothing downstream shifts under it.

Published baselines: AR, Pocket2Mol, DiffSBDD, DecompDiff, FuncBind.
Reference = the deposited crystal ligand, re-scored through the same Vina protocol.
Ours: VoxBind-Ours (= CoDE, density-conditioned) and Ours-v2; VoxBind / VoxBind-vanilla
are the σ=0.9 vanilla rows (ep923 on the reporting box, published ckpt at 100/pocket here).
svr12 8-GPU arms: VoxBind-base-ep350 (+ `-n100`, the same arm at 100 molecules/pocket),
Fusion-v4-cdgv2-{warm-ep100,scratch-ep350}, Fusion-v4-cv2-scratch-ep350,
Fusion-default-cv2-scratch-ep350.

`samples/` present for DecompDiff (reproduced `.pt`), VoxBind + Ours-v2 (per-target
viz `.sdf`). AR / Pocket2Mol / DiffSBDD / FuncBind carry `SOURCE.txt` — their
generated samples live on the baselines server (`prj-denovo/baselines/`), not here.

### `<method>/eval/` — one folder per evaluation

task2 was evaluated on three different boxes and the numbers used to sit in four
different shapes: per-molecule `target_*/metrics.json` trees for the svr12 arms,
`_shared/baselines_eval/*.json` arrays for the published baselines, `_shared/260910_*`
for the figure slices, and a CSV under `notebook/html` for rigid-fragment. Same arm,
four names, and nothing said which evaluations a method actually had.

`voxbind/scripts/tools/collect_task2_eval.py` reads every one of those sources and
writes them out in one shape. It only ever **copies** — nothing moves out of
`voxbind/exps/`, and no `<method>/metrics.json` is touched (the reports read those).

```
<method>/eval/
├── index.json                    which evaluations exist, their source + host + counts
├── vina_docking/                 Vina score_only / minimize / dock, high-affinity %
├── sample_quality/               validity, uniqueness, diversity, QED, SA, logP, Lipinski
├── posebusters/                  dock-mode validity + the 20 per-check pass rates
├── posecheck/                    steric clashes, torsional strain, interaction profile
└── rigid_fragment/               rigid-fragment RMSD vs. fragment size
      each holding  results.json       the aggregate, per pocket set
                    per_molecule.csv   one row per molecule  (per_fragment.csv for rigid)
```

**The per-molecule tables are the point** — ~330k rows, one per (pocket, generated ligand).
Every table opens with the same four key columns:

```
target,pocket_index,in_density79,ligand_filename,mol_idx,smiles,n_atoms,<metric columns…>
target_00,0,0,BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf,0,OS(O)(O)O,5,-2.4,…
```

`ligand_filename` is the pocket's CrossDocked identity, so a row can be traced to the test
set and to the receptor PDB without this bundle's `target_NN` ordinal. `smiles` and
`n_atoms` are on every row of every table, PoseBusters included, so size-resolved questions
work the same way everywhere. Joining on `target` + `mol_idx` gives one frame per method,
which is what makes QED-against-Vina, the strain tail, or "which pocket did the PoseBusters
failures come from" answerable without going back to the boxes. Two things had to be right
for that join to be trustworthy:

* **The index is the meta order, not the directory order.** The baselines' `_eval` dumps
  are split across `<M>`, `<M>_part2` and `<M>_gap`; walking those in order gives a
  different sequence from the one the pose scorers used, and joining on it pairs the wrong
  molecules (measured before the fix: 1% SMILES agreement on DiffSBDD). The meta bundles in
  `<method>/samples/meta/` are the canonical order — their totals *are* the published
  scored counts — so they supply the index and the membership, and `_eval` supplies only
  values, matched on `(SMILES, rounded-coordinate hash)`.
* **The join is verified, not assumed.** Each `results.json` records
  `per_molecule.cross_table_smiles_agreement`, the measured SMILES agreement between every
  pair of tables, and `cross_table_join` says `safe` or `UNSAFE`. 12 of 13 methods are safe;
  `VoxBind-Ours` is not — its docking pass and its pose pass were run months apart on
  different orderings, so join that one on `(target, smiles)`.

Missing values are written as empty cells, never `nan`: one `nan` string is enough to make
every median a reader computes silently wrong.

Every `results.json` carries the same envelope — `method`, `folder`, `evaluation`,
`generated_on`, `evaluated_on`, `source`, `protocol`, then `pocket_sets` with an `all`
and/or a `density79` slice. Two axes are always named apart: `qed_mean` pools molecules,
`qed_mean_over_pockets` averages per-pocket means (they differ when pockets yield unequal
counts), and PoseCheck's `strain_mean_UNRELIABLE` says in its name why not to quote it.

```bash
python3 voxbind/scripts/tools/collect_task2_eval.py            # write / refresh
python3 voxbind/scripts/tools/collect_task2_eval.py --check    # re-derive & verify only
python3 voxbind/scripts/tools/collect_task2_eval.py --dry-run  # show what would change
```

Run it under an interpreter with torch + rdkit (`~/miniforge3/envs/sbdd/bin/python`) or the
baselines' per-molecule tables, which come out of `.pt` dumps, are skipped — loudly, not
silently. Otherwise re-runnable on any box that has the bundle (no `/home1/irteam` paths),
and a file whose payload has not changed keeps its mtime, so `dropbox_push.sh` skips it
instead of re-uploading. `EVAL_STATUS.md` is the method x evaluation matrix, the
per-molecule row counts, and the list of runs generated but never fully evaluated.

## task3-mcp  (FuncBind macrocyclic peptides, pilot)

`metrics.json` = the values from `reports/results_mcp.html`. Methods:
FuncBind-vanilla, Ours-receptorED, Reference. FuncBind `.sdf/.pdb` samples are
split by run-name heuristic (vanilla eval runs → FuncBind-vanilla; curated
paper-100 → Ours-receptorED; one ambiguous paper run in `_shared/`).

## Excluded on purpose — checkpoints & configs

Model weights/configs are **not** duplicated here — they live in
`voxbind/model_zoo/` (+ per-run `voxbind/exps/`), backed up to Dropbox separately
at `박성현/VoxBind/model_zoo`. Also not copied: multi-GB raw per-sample eval
(`base/_casf/*.csv`, `base/decompdiff/eval_results_refprior25/*.pt`).

## docker/

`Dockerfile` + `env.yaml` / `env.minimal.yaml`, copied from the repo root, so the
bundle carries its own runnable environment.

## Backup (Dropbox, via rclone)

```bash
bash results/dropbox_push.sh        # local -> Dropbox (박성현/VoxBind/results)
bash results/dropbox_pull.sh        # Dropbox -> local (entire bundle)
bash results/dropbox_pull.sh VoxBind-Ours  # Only the current reported Ours model
bash results/dropbox_pull.sh VoxBind VoxBind-Ours  # Multiple models
bash results/dropbox_pull.sh --task task2-drugdesign --list  # Discover model names
bash results/dropbox_pull.sh VoxBind-Ours --dry-run  # Preview only
```

Prereq: an rclone remote named `dropbox` with `root_namespace_id = 12221840097`
(see `notebook/html/dropbox-sync.md`). `rclone copy` is incremental and resumable.

### Sample-only pulls — `dropbox_pull_baselines.sh`

The full `dropbox_pull.sh` is 3.8 GiB, and 3.47 GiB of that is one folder:
`DecompDiff/samples/outputs_*/`, the raw generator dump. The molecules every
analysis actually reads are the 12.7 MiB of `samples/meta/*.pt` beside it. So for
"just give me the baselines' samples" there is a narrow pull:

```bash
bash results/dropbox_pull_baselines.sh                   # the 5 CrossDocked baselines, ~68 MiB
bash results/dropbox_pull_baselines.sh AR DiffSBDD GET   # any number of methods, any task
bash results/dropbox_pull_baselines.sh -l                # list the methods on the remote
bash results/dropbox_pull_baselines.sh --with-eval        # + per-pocket vina/posecheck/posebusters
bash results/dropbox_pull_baselines.sh -n DecompDiff     # dry run
bash results/dropbox_pull_baselines.sh -a --task task3-mcp --with-shared
```

Per method it takes `samples/**` + `metrics.json` + `SOURCE.txt`, and skips
`run/` (cfg, hydra, train logs), `representations/` (task1 cached features),
`eval/` (the per-pocket vina / posecheck / posebusters behind `metrics.json` —
add it with `--with-eval`) and — unless `--with-raw` — `samples/outputs_*/`. Method names resolve across all three
tasks, so pass a bare name; `--task` disambiguates when one exists in two tasks.
It prints a per-method size table before transferring anything.

Note the filters use `--filter`, not `--include`/`--exclude`: rclone parses those
two in an **indeterminate order** (it warns about it), and the `outputs_*`
exclusion loses to the `samples/**` inclusion often enough that the "small" pull
silently becomes the 3.5 GiB one. `--filter` rules apply strictly in order.

## Current reported VoxBind + Ours: full sample bundle

`task2-drugdesign/VoxBind-Ours/` contains the experiment used as **VoxBind + Ours**
in the current qualitative notebook and 260910 figures:

- Run: `voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9`, epoch 350.
- Samples: `samples/full_eval_ep350`, all 79 density pockets.
- The bundle preserves generated `samples.sdf`, reference ligand SDFs, pocket10
  PDBs, per-target `metrics.json`, and all root-level evaluation JSONs.
- Headline evaluation: `samples/eval_docking_results_full79.json`
  (full receptor, 79 pockets; target-averaged Vina Dock **-8.488612 kcal/mol**).
  Crop/full/rerun JSONs remain separate and retain their original filenames.
- `metrics.json` records the source run, selected evaluation protocol, counts,
  and the original summary. `SOURCE.txt` explains provenance and exclusions.
- `SHA256SUMS` verifies the bundled files: run `sha256sum -c SHA256SUMS`
  inside the method folder.
- Generated SDF poses are not redocked poses. Full-receptor PDB inputs and
  checkpoints are not included; pocket10 PDBs support the qualitative view.

The older remote `Ours-v2/` folder is left untouched. It contains headline
metrics and only three visualization SDFs, not this complete 79-pocket bundle.
Its folder name must not be used to infer the identity of a local Ours v2 run.

### Model-selective pulls

`dropbox_pull.sh` accepts one or more model folder names (case-insensitive),
or repeated `--method MODEL` / `--model MODEL`. It resolves all names before
copying, reports missing/ambiguous names, and supports `--task` for
disambiguation. For example:

```bash
bash results/dropbox_pull.sh --method VoxBind-Ours
bash results/dropbox_pull.sh --task task2-drugdesign VoxBind-Ours
RESULTS_DEST=/data/results bash results/dropbox_pull.sh VoxBind-Ours
bash results/dropbox_pull.sh VoxBind-Ours -- --checksum --transfers 8
```

With no selectors, the script still downloads the entire results bundle.
`--task TASK` without model names downloads that task, including shared
artifacts. Selectors must precede extra rclone flags; use `--` to separate them.
A model-only pull copies the complete model folder, including evaluation
JSONs and provenance, but does not touch other models. The existing
`dropbox_pull_baselines.sh` remains the smaller sample-only alternative.
All transfers use `rclone copy`, never `sync` or deletion.
