# Boltz-2 affinity baseline (zero-shot, pretrained)

Evaluate the **pretrained Boltz-2 binding-affinity module** zero-shot on our
`lp_edrscc_v2` **test** split, as a reference baseline for the VoxBind affinity task.
Boltz-2 (Wohlwend et al., 2025; [boltz.bio/boltz2](https://boltz.bio/boltz2),
[paper](https://jeremywohlwend.com/assets/boltz2.pdf)) is an AF3-style co-folding model
with an added affinity head that, from **protein sequence + ligand SMILES**, predicts a
3D complex and then a binding affinity — no experimental structure needed.

No training here: we only run inference with the released weights.

## Can we test it on our split? — Yes, with three big caveats

1. **Data leakage / not held-out.** Boltz-2's affinity module is trained on public
   Kd/IC50 assays + PDB structures; `lp_edrscc_v2` is PDBbind, which almost certainly
   overlaps Boltz-2's training set (pocket-level leakage has been flagged publicly). So
   any number here is **in-distribution for Boltz-2**, not a clean held-out test like our
   frozen probes — it will *over*-state Boltz-2's generalization.
2. **Designed for within-series ranking, not cross-target absolute affinity.** The authors
   state the pIC50 output "is not supposed to be used on arbitrary chemical spaces but
   only for hit-to-lead stage compound series." Our test set spans ~859 distinct targets,
   so weak/poor correlation is expected and does *not* reflect its intended use.
3. **Unit mismatch.** Boltz-2 predicts `log10(IC50 µM)`; our labels are pKd/pKi. We map
   `pred_pK = 6 − affinity_pred_value` (≈ pIC50) — read the **correlation** (Spearman),
   not RMSE.

## Pipeline

```
src/make_inputs.py   # split+labels -> one Boltz YAML per test complex (protein seq + ligand SMILES + affinity)
src/run_boltz.sh     # boltz predict --use_msa_server  (needs `boltz` env + GPU; weights auto-download)
src/score.py         # gather affinity_*.json -> Pearson/Spearman/RMSE vs pK + binder-prob ranking
```

Boltz-2 affinity limits applied in `make_inputs.py`: ligand ≤ 128 atoms (incl. H) is a
hard drop; > 56 heavy atoms is kept but flagged; long sequences guarded by `--max_seq_len`.
The `input_manifest.csv` records the evaluable universe and every skip reason (no silent
truncation). On the 1320 test complexes: 1311 have SMILES, ~46 ligands exceed 128 atoms,
~859 unique targets need MSAs.

## Cost

Each complex co-folds (trunk + structure + affinity ensemble) on GPU — roughly minutes
each, plus MSA generation per unique target. Full test (~1250 evaluable) is tens of
GPU-hours; start with a subsample (`make_inputs.py --limit N`) for a first read.

## Reproduce

```bash
# 1) prepare inputs (rdkit+pandas only)
python src/make_inputs.py --msa server          # add --limit 200 for a subsample
# 2) predict (boltz env + GPU)
CUDA_VISIBLE_DEVICES=<gpu> bash src/run_boltz.sh
# 3) score
python src/score.py --tag edrscc
```
