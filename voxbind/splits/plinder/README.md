# Frozen PLINDER pretraining selection (versioned)

Which ligand-instances make up the PLINDER ligand-matched density pretraining corpus, pinned
cross-server. Layout is **versioned** — one subdir per version, each self-contained:

```
splits/plinder/
  v1/   plinder_selected.csv  plinder_inputs.json  plinder_funnel.json
  v2/   plinder_selected.csv  plinder_inputs.json  plinder_funnel.json   (when built)
  .gitignore  README.md
```

Each version dir holds: `plinder_selected.csv` (one row per kept ligand-instance), `plinder_inputs.json`
(pinned bucket / filters / build params + sha256 of the selection and the leakage holdout files),
and `plinder_funnel.json` (full provenance funnel).

| version | n instances | key filters |
|---|---|---|
| **v1** | 17,721 | `max_res 2.5`, single-ligand, PLINDER `train` split only, `dedup=pdb`, RSCC≥0.8, no cofactors |
| **v2** | ~10–11K | **no resolution cap** (RSCC≥**0.95** gates), all splits, multi-ligand, `dedup=none`, cofactors admitted, **`cap_per_ccd 1`** (one best-density example per ligand chemotype), **in-vocab C/O/N/S/F/Cl/P filter**. Strict + maximally diverse small set; median res 1.98 Å, median RSCC 0.968. |

## Consumed by
`dataset/legacy/03b,03c` (and the `build/` pipeline) via `voxbind.splits.frozen_plinder_selection(version=...)`,
which resolves `splits/plinder/<version>/`, verifies the sha256, and loud-fails on drift. Default version is
`DEFAULT_PLINDER_VERSION` (`v2`); it falls back to `v1/` (then the legacy flat dir) with a warning if absent.
`03b/03c` also take `--plinder_version`.

## Re-pin / build a version
Deliberate, version-bumped step. v2 recipe (run where the **proper leakage files** live —
`pdbbind/index.csv` *with* a `new_split` column, `misato/pool_pids.txt`, `data_test.pt`):

```bash
cd voxbind
python dataset/legacy/03a_plinder_select.py \
    --max_res 0 --min_rscc 0.95 \
    --no-single_ligand --plinder_splits --dedup none \
    --allow_cofactor --cap_per_ccd 1 --vocab_filter \
    --version v2 --rederive --freeze
```

Scale the size with two knobs: lower `--min_rscc` (0.95→0.9 roughly doubles it) and/or raise
`--cap_per_ccd` (allow N poses per chemotype). The recipe above targets a strict, maximally
diverse ~10K set.

`--freeze` writes `splits/plinder/v2/` and sets `name=plinder_v2`. v1 stays loadable alongside.
Without the proper `pdbbind/index.csv`, 03a SKIPS pdbbind leakage exclusion with a loud warning —
the resulting split is **not** leakage-safe and must not be frozen for real use.
