# PLINDER v2.5 — LP-PDBBind CL3-test ID30-clean

PLINDER v2.5 is the existing per-element PLINDER-v2 pretraining corpus after
removing every ligand observation whose PDB contains a protein chain similar to
any chain in the `lp_edrscc_v2_cl123` **test** cohort (733 proteins) — the same
test set scored by the ID60/ID30 affinity-novelty tables. It is the direct
sibling of [v2.4](../v2p4/README.md), which cleans against CASF-2016; the policy
and mechanism are identical so the two corpora are comparable.

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
| Removed | 16,441 (14.6%) | 7,765 (18.6%) |
| v2.5 retained | 96,292 | 33,912 |
| Training / validation | 96,192 / 100 | — |

The frozen selection CSV contains 97,423 retained ligand instances from 34,068
PDBs (slightly larger than the loader manifest because the v2 tuple build applies
its own structure/size and reserved-tail filters).

The authoritative training view is
`dataset/data/pretrain/xray_resample_plinder_v2p5_perelem/train_manifest.npz`.
Its `ok` mask exposes only retained rows. `box116.dat` is a hard link to the
position-identical v2 density box, so v2.5 adds no second 164 GB allocation.

## How this compares to the MMseqs2 audit

An earlier audit measured CL3-test overlap with MMseqs2 under the stricter
*both*-sequence 80% coverage rule (`cov-mode 0`), which removed 12.4% of
positions / 16.2% of PDBs. v2.5 instead uses v2.4's *shorter*-sequence coverage
rule (more aggressive: it also removes a large PLINDER protein whose domain
matches a smaller test chain), giving the 14.6% / 18.6% above. The shorter-
coverage rule was chosen for exact parity with v2.4.

## Build and validate

```bash
python dataset/plinder/02c_make_v2p5_cl3test_clean.py --threads 24
python dataset/plinder/validate_v2p5.py --loader-smoke
```

The reusable dataset configuration is `configs/dset/plinder_v2p5_box.yaml`.
Per-PDB removal witnesses are in `cl3test_id30_matches.tsv`, the exact removed
PDB list is in `cl3test_id30_removed_pdbs.txt`, and the frozen query cohort is
`cl3test_cohort.fasta`.
