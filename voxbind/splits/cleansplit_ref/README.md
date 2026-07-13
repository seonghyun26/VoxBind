# CleanSplit reference (GEMS)

Vendored inputs + filtering scripts for the **PDBbind CleanSplit** from:

> Gao, Durairaj et al., *"Resolving data bias improves generalization in binding affinity
> prediction"*, **Nature Machine Intelligence 2025**
> ([nature](https://www.nature.com/articles/s42256-025-01124-5) ·
> [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.09.627482) ·
> code [camlab-ethz/GEMS](https://github.com/camlab-ethz/GEMS)).

CleanSplit is a structure-based filtering of the PDBbind training set that removes **both**
(1) train↔test leakage into the CASF benchmarks and (2) **redundancy within the training
set** — the second being what LP-PDBBind (and our `lp_edrscc_*`) do not do.

## Files

- `PDBbind_data_split_cleansplit.json` — the released cleaned lists:
  `train` (16,908), `casf2016` (285), `casf2016_indep` (144, the leakage-independent CASF
  subset), `casf2013`, `casf2013_indep`.
- `PDBbind_cleansplit_train_val_split_f0.json` — fold-0 `train` (13,192) / `validation` (3,299).
- `remove_train_test_sims.py` / `remove_train_redundancy.py` — the GEMS filtering algorithm
  (uses pairwise TM-score / Tanimoto / ligand-RMSD / affinity-difference thresholds). Needs
  the precomputed similarity matrices from [Zenodo](https://doi.org/10.5281/zenodo.14260170)
  to run — only required to **re-derive** CleanSplit with custom thresholds; our
  `clean_ed_*` schemes use the released lists above directly.
- `dataset_filtering.md` — GEMS' own description of the algorithm.

## How this feeds our splits

`voxbind/splits/make_splits.py::build_clean_ed()` intersects these released lists with our
eligible universe (ED-available ∧ lig&poc RSCC ≥ 0.8 ∧ Kd/Ki, non-covalent — same bar as
`lp_edrscc_v2`) to produce the `clean_ed_v1` / `clean_ed_v1_indep` schemes. Not yet
generated — see `../README.md`.
