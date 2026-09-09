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
├── task2-drugdesign/   <method>/  samples/          metrics.json   (+ SOURCE.txt)
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

`metrics.json` = PoseCheck aggregate (strain/clash/heavy-atoms from the local
per-method posecheck JSONs) and, for the voxel-diffusion models, Vina Dock over
the 79 density pockets. Methods: AR, Pocket2Mol, DiffSBDD, DecompDiff, FuncBind,
VoxBind (σ=0.9 vanilla), Ours-v2 (density-conditioned).

`samples/` present for DecompDiff (reproduced `.pt`), VoxBind + Ours-v2 (per-target
viz `.sdf`). AR / Pocket2Mol / DiffSBDD / FuncBind carry `SOURCE.txt` — their
generated samples live on the baselines server (`prj-denovo/baselines/`), not here.

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
bash results/dropbox_pull.sh        # Dropbox -> local
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
bash results/dropbox_pull_baselines.sh -n DecompDiff     # dry run
bash results/dropbox_pull_baselines.sh -a --task task3-mcp --with-shared
```

Per method it takes `samples/**` + `metrics.json` + `SOURCE.txt`, and skips
`run/` (cfg, hydra, train logs), `representations/` (task1 cached features) and —
unless `--with-raw` — `samples/outputs_*/`. Method names resolve across all three
tasks, so pass a bare name; `--task` disambiguates when one exists in two tasks.
It prints a per-method size table before transferring anything.

Note the filters use `--filter`, not `--include`/`--exclude`: rclone parses those
two in an **indeterminate order** (it warns about it), and the `outputs_*`
exclusion loses to the `samples/**` inclusion often enough that the "small" pull
silently becomes the 3.5 GiB one. `--filter` rules apply strictly in order.
