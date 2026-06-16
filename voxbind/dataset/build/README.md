# Unified dataset-build pipeline (`dataset/build/`)

**One file per pipeline *stage***, each with a `--dataset {crossdocked,pdbbind,misato,plinder}`
if-branch — instead of one script per dataset. Replaces the per-dataset sprawl
(`01a/01b`, `02a–d`, `03a–c`) for PDBbind / MISATO / PLINDER.

> **CrossDocked is left intact.** `00a`, `00b`, `00f`, `00h`, `preprocess_crossdocked.py`,
> `crossdocked.py`, `crossdocked_xray.py` are **not moved or modified**. The `crossdocked`
> branch here **delegates** to those existing, validated scripts.

## Decisions (260616, user-confirmed)
- **Scope:** all four datasets unified behind these stages.
- **Voxelize = hybrid:** dense voxels precomputed to disk **only for the downstream probe
  sets** (PDBbind, MISATO). Pretraining (CrossDocked, PLINDER, v7) keeps the compact
  `(pocket,ligand)` point-cloud tuples + **on-GPU voxelization at train time** — that point
  cloud *is* the precompute; dense voxels for 78k CrossDocked would be ~450 GB and would kill
  rotation augmentation.
- **Migration:** PDBbind/MISATO/PLINDER logic is **extracted** into stage branches (the subtle
  bits — PLINDER `label_asym_id` parsing, RSCC gate — are *moved, not rewritten*). Originals go
  to `dataset/legacy/` only **after numerical parity** is confirmed. CrossDocked originals stay.

## Stages
| stage | file | does (per-dataset branch) |
|---|---|---|
| 1 | `s1_acquire.py` | download/locate originals; select+dedup+leakage (PLINDER); element→channel map + **mask** unsupported atoms (keep complex); size filter; **split assign**; emit `(pocket,ligand)` tuples + manifest |
| 2 | `s2_density.py` | download CCP4/EDS for **matched** entries; per-dataset **match gate**; resumable + stream-and-discard |
| 3 | `s3_crop.py` | **align** → crop box at ligand centroid → **normalize** (canonical arcsinh+z) → **availability gate** → gradmag; emit crops + `*_available.npy` + stats.json |
| 4 | `s4_voxelize.py` | **hybrid**: precompute dense voxels for probe sets; no-op (point clouds) for pretraining |
| 5 | `s5_targets.py` | downstream labels: PDBbind pK · MISATO QM/MD · B-factors |
| — | `registry.py` | per-dataset config (below) + the VoxBind atom dictionary |

## Per-dataset config (`registry.py`)
| dataset | frame → align | density match gate | split source | stage-4 | targets |
|---|---|---|---|---|---|
| **crossdocked** | docking → crystal (**Kabsch**) | `native_filter tt_min` + density-at-atoms | val-from-train-tail | on-the-fly | none (SSL) · **delegates to `00*`** |
| **pdbbind** | deposited (identity) | inherent holo (PDBe-EDS coverage) | `new_split` 2172/480/839 | **precompute** | pK from `index.csv` |
| **misato** | deposited (identity) | built structures + QM/MD coverage | MISATO 8:1:1 | **precompute** | QM/MD from hdf5 |
| **plinder** | deposited (identity) | **RSCC ≥ 0.8** + 0 unresolved heavy | val-from-train-tail | on-the-fly | none (SSL) |

## Atom-blob radii (stage 4 + train-time voxelizer)
Atom channels rasterize as blobs whose radius is set **independently for ligand and pocket**,
each with two modes (already supported in `crossdocked_xray.py.__getitem__` via the
`*_radius` / `*_radius_lut` branches):
- **uniform** — one scalar radius for every atom (`ligand_radius` / `pocket_radius` = `R > 0`, e.g. 0.5 Å)
- **vdW** — per-element van der Waals radius via LUT (`ligand_radius` / `pocket_radius` = `-1`)

→ four combinations from two registry knobs: `ligand_radius ∈ {R, vdW}` × `pocket_radius ∈ {R, vdW}`.
The **same** knobs drive the on-GPU train-time voxelizer (pretraining) **and** the precomputed
probe voxels (stage 4) — they must match or frozen features won't line up.

> **Default = vdW for both** (`ligand_radius = pocket_radius = -1`, "ligvdw"). Mixing modes is
> fine for *separate* lig/poc channels, but in a *merged* lig+poc channel a uniform-ligand /
> vdW-pocket mix leaks ligand-vs-pocket identity by blob **size** (the 260603 merged-blob fix).

## Run
```
# each stage takes --dataset {crossdocked,pdbbind,misato,plinder}; add --dry_run to preview
python dataset/build/s1_acquire.py  --dataset pdbbind        # 01a structures + index
python dataset/build/s2_density.py  --dataset pdbbind        # 01a density (PDBe-EDS)
python dataset/build/s4_voxelize.py --dataset pdbbind        # 01b voxelize+poolnorm (crop fused in; s3 is a no-op)
python dataset/build/s5_targets.py  --dataset pdbbind        # pK is in index.csv; bfactors via extract_bfactors
```

## Wiring notes (current: branches delegate to the NN_ scripts via subprocess)
- **PDBbind** — s1→`01a structures+index`, s2→`01a density`, s4→`01b voxelize+poolnorm` (crop fused), s5: pK lives in `index.csv`.
- **PLINDER** — s1→`03a` select, s2→`03b --what both` (cif+ccp4), s3→`03c` (crop+normalize+tuples). `03c` uses gemmi → the dispatcher sets `LD_LIBRARY_PATH=<interp>/lib` automatically (CXXABI).
- **MISATO** — **dual-env**: `02a` (acquire) & `02d` (voxelize+feature-extract, *fused*) run under voxbind; **`02b` targets run under an h5py interpreter** (`MISATO_H5PY_PY`, default cellmaes). PDBbind-missing complexes must first be built via `02c_misato_structures.py build --ids <list>` (h5py) — noted in s3, not auto-run (needs an id list).
- **CrossDocked** — delegated to `00a`/`00b`/`preprocess_crossdocked` (unchanged).
- Passthrough forwards to **every** step in a stage → for multi-step stages use only common flags; step-specific flags (e.g. `--ligand_vdw`) are registry-driven.

## Validation (before retiring any original)
Regenerate a handful of samples through the new stages and assert **numerical equivalence**
against the current outputs (crops, tuples, voxels) per dataset. Only then move that dataset's
old scripts to `dataset/legacy/`. CrossDocked is delegated, so nothing to validate there.
