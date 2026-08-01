# VoxBind v4 pocket-density corpus

V4 keeps the established VoxBind PLINDER training contract while adding
leakage-safe apo states:

- tuples: `dataset/data/pretrain/data_train_plinder_v4_perelem.pt`;
- positional OTF manifest: `dataset/data/pretrain/xray_resample_plinder_v4_perelem/`;
- shared experimental maps: `dataset/data/ccp4/{pdb_id}.ccp4`;
- provenance: `dataset/data/v4/manifest.parquet` and `pair_edges.parquet`.

Each tuple is the existing `(pocket_dict, ligand_dict)` structure. Holo tuples
contain the ligand; apo tuples set `ligand_present=false`, so the loader emits
zero ligand atoms while retaining the transferred holo anchor as
`center_coords`. Pockets use the exact VoxBind atom rule: supported protein
heavy atoms no farther than 10 Å from the holo ligand. No whole-residue
expansion is used.

Paired states are rewritten into one deterministic pocket-local coordinate
frame. The positional manifest's `R` and `t` map that frame back to each
structure's own deposited experimental map. Apo density therefore always comes
from the apo crystal, never from its holo partner.

## Rebuild

```bash
python dataset/v4/build_ahoj_local.py --jobs 4
python dataset/v4/build_corpus.py
python dataset/v4/acceptance.py dataset/data/v4/manifest.parquet \
  --pairs dataset/data/v4/pair_edges.parquet \
  --density-root dataset/data --check-density-files \
  --coverage-report dataset/data/v4/coverage_report.md
```

`build_corpus.py` extends PLINDER's
`pocket_fident__50__weak__component`, propagates the protected affinity and
CrossDocked2020 test PDBs, unions exact UniProt/PDB/pair/interface relations,
and enforces PLINDER matched molecular series from
`mmp/plinder_mmp_series.parquet`. Split precedence is `test > val > train`.

## Train

```bash
torchrun --standalone --nproc_per_node=4 train_density.py \
  --config-name=config_train_atomblob8_density_gradmag_channelvit_mae_40m_plinder_v4_otf
```

The Hydra configuration remains overrideable, for example
`bsz=4 accum_steps=8 num_epochs=50 mae.mask_ratio=0.75`.
