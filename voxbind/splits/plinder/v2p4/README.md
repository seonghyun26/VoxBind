# PLINDER v2.4 — CASF-2016 ID30-clean

PLINDER v2.4 is the existing per-element PLINDER-v2 pretraining corpus after
removing every ligand observation whose PDB contains a protein chain similar to
any chain in the local 214-complex CASF-2016 evaluation cohort.

## Policy

- DIAMOND 2.1.8 sensitive protein search.
- Sequence identity at least 30%.
- Alignment covers at least 80% of the shorter sequence, implemented as
  `max(query_coverage, subject_coverage) >= 80`.
- If any chain qualifies, all ligand observations from that PLINDER PDB are
  removed.

## Frozen counts

| Stage | Observations | Unique PDB |
|---|---:|---:|
| v2 loader manifest | 112,733 | 41,677 |
| Removed | 11,526 | 6,087 |
| v2.4 retained | 101,207 | 35,590 |
| Training / validation | 101,107 / 100 | — |

The earlier frozen selection CSV contains 102,376 retained ligand instances
from 35,746 PDBs.  Its count is slightly larger than the loader manifest because
the v2 tuple build applies its own structure/size and reserved-tail filters.

The authoritative training view is
`dataset/data/pretrain/xray_resample_plinder_v2p4_perelem/train_manifest.npz`.
Its `ok` mask exposes only retained rows.  `box116.dat` is a hard link to the
position-identical v2 density box, so v2.4 adds no second 164 GB allocation.

## Build and validate

```bash
python dataset/plinder/02b_make_v2p4_casf_clean.py --threads 24
python dataset/plinder/validate_v2p4.py --loader-smoke
```

The reusable dataset configuration is `configs/dset/plinder_v2p4_box.yaml`.
Per-PDB removal witnesses are in `casf_id30_matches.tsv`, and the exact removed
PDB list is in `casf_id30_removed_pdbs.txt`.

