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
| **v2.4** (`v2p4`) | 102,376 frozen selections; 101,207 loadable positions | v2 decontaminated against all 214 local CASF-2016 evaluation complexes at ID≥30% and ≥80% coverage of the shorter protein sequence. The loader exposes 101,107 train + 100 validation positions and reuses the position-identical v2 density box. |

## Consumed by
`dataset/legacy/03b,03c` (and the `build/` pipeline) via `voxbind.splits.frozen_plinder_selection(version=...)`,
which resolves `splits/plinder/<version>/`, verifies the sha256, and loud-fails on drift. Default version is
`DEFAULT_PLINDER_VERSION` (`v2`); it falls back to `v1/` (then the legacy flat dir) with a warning if absent.
`03b/03c` also take `--plinder_version`.

## Linked experimental apo pockets

`dataset/plinder/06_build_apo_pairs.py` intersects any frozen `v2`, `v2p1`, or
`v3` selection with PLINDER's experimental apo links. It keeps only local apo
PDBs with an existing 2Fo-Fc map, aligns the holo binding site into the apo map
frame, defines the apo pocket around that site, and writes paired records plus a
model-ready apo tuple/resampling manifest. The aligned holo ligand is an
explicit crop anchor, not an input: apo tuples carry `ligand_present=false`, so
the dataset loader emits zero ligand atoms.

The current local v2 snapshot (2026-07-30) has 2,964 density-backed candidate
holo systems after selecting the best apo link per system. Strict coordinate,
density, and no-nearby-organic-ligand validation retains **1,329 apo pockets
from 179 unique apo PDB/map targets**. The standard deterministic loader split
contains 1,227 train and 100 validation samples after the 30 Å size filter.

```bash
cd voxbind

# Count local candidates without building coordinates.
python dataset/plinder/06_build_apo_pairs.py --version v2 --selection-only

# Build/validate paired records, apo-only tuples, and train/val density manifests.
python dataset/plinder/06_build_apo_pairs.py --version v2
```

Ignored generated outputs live under `dataset/data/pretrain/`:

- `plinder_v2_apo_pairs.pt`: paired holo/apo records and provenance;
- `data_train_plinder_v2_apo.pt`: model tuples with no ligand atoms;
- `plinder_v2_apo/{selected_links,pair_index,skipped}.csv`;
- `xray_resample_plinder_v2_apo/`: train/val manifests and map symlinks.

The interactive paired-state viewer is
`notebook/html/260806/visualize_plinder_apo_pockets.ipynb`. It shows the apo
pocket and apo experimental density, then the matching deposited-frame holo
pocket, real ligand, and its separate experimental density map. A clearly
labelled visualization-only "dry pocket" panel identifies modeled apo waters
and smoothly masks their local density without modifying the source CCP4 map or
the training data.

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
