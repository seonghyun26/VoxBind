# `base/` — external baselines

Self-contained third-party (and re-implemented) models we compare VoxBind against.
Each subfolder is a standalone project: it imports nothing from `voxbind/`, brings its
own environment, and reads only shared data under `voxbind/dataset/data/`.

**Conventions**
- Untracked in git; heavy outputs (weights, data, logs, preds) stay local via `.gitignore`.
- Folders are referenced **by path** (`base/<name>`) from `voxbind/` and `notebook/` —
  **don't rename or move them** or those references break.
- Per folder: `README.md` = upstream project readme; `README_edrscc.md` = **our**
  integration on the `lp_edrscc_v2` split (present where the upstream repo was vendored).
  The `_edrscc/` subdir holds our glue (data conversion, train/eval on our split).

## Affinity baselines

Comparators for the VoxBind affinity task. ρ below is the quick-reference **CASF-2016
(leaky) Spearman, n=214**, from `_casf/<Model>.json`; see *Where the numbers live* for
the authoritative, full tables (lp_edrscc_v2 + CASF-clean, all three metrics).

| folder | model · venue | how run | ρ | status |
|---|---|---|---|---|
| `cheapnet` | CheapNet · ICLR 2025 | trained on split | 0.836 | done |
| `profsa` | ProFSA · ICLR 2024 | pretrained encoder + probe | 0.749 | done |
| `ipdiff` | IPDiff/**IPNet** · ICLR 2024 | frozen interaction-prior features + probe | 0.765 | done (`IPNet_frozen`; also `IPNet_retrain` 0.675) |
| `hbgsa` | HBGSA · arXiv (reimpl) | trained on split | 0.696 | done |
| `aevplig` | AEV-PLIG · Comms Chem 2025 | trained on split | 0.691 | done |
| `get` | GET · ICML 2024 | trained on split | 0.678 | done |
| `dsmbind` | DSMBind · NeurIPS 2023 | unsup. energy; frozen feats + probe | 0.677 | done |
| `nesso` | Nesso-1 · Valence/Recursion 2026 | zero-shot cofolding (PDBbind-leaked) | 0.672 | done |
| `honestaffinity` | HonestAffinity · arXiv (reimpl from spec) | trained on split | 0.661 | done |
| `bindnet` | BindNet · ICLR 2024 | trained from scratch (weights restricted) | 0.458 | done (weak) |
| `boltz2` | Boltz-2 · 2025 | zero-shot cofolding + affinity head | — | harness built, **not yet run** |

`EGNN` / `EGNN_TD` (`_casf/EGNN*.json`, ρ 0.688 / 0.684) are simple GNN references
computed inline — no dedicated folder.

## Generative (SBDD) baselines

3D structure-based molecule generation — sampling + docking eval, not affinity regression.

| folder | model · venue | role |
|---|---|---|
| `decompdiff` | DecompDiff · ICML 2023 | diffusion SBDD; ref-prior / improved sampling + Vina eval |
| `ipdiff` | IPDiff · ICLR 2024 | diffusion SBDD sampling (its IPNet also serves the affinity table above) |

## Shared / auxiliary

| path | what |
|---|---|
| `_casf/` | aggregated CASF-2016 result JSONs (one per model, incl. our `C`/`CDG*`); written by `voxbind/test/probe_casf.py`. Keep — consumed by the notebook result builders. |
| `_cl_campaign/` | logs + launch scripts from the CASF-clean benchmark campaign (job orchestration; ephemeral). |
| `clean_ed_results.md` | `clean_ed_v1` (CASF-2016 CleanSplit) results summary: ours vs ProFSA vs GET, with caveats. |
| `.gitignore` | shared ignore rules keeping baseline outputs local. |

## Where the numbers live

- **`notebook/html/results.html`** — canonical VoxBind Table 1 (lp_edrscc_v2, ours + all baselines).
- **`clean_ed_results.md`** — CASF-2016 CleanSplit head-to-head (ours / ProFSA / GET).
- **`_casf/*.json`** — raw per-model CASF-2016 scores (leaky n=214 + non-train), all three
  metrics (Pearson / Spearman / RMSE) with seed std.
