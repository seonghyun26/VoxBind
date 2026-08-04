# Nesso-1 Baseline

**Model**: Nesso-1 (Valence Labs / Recursion, 2026)  
**Repo**: https://github.com/recursionpharma/nesso  
**Weights**: https://huggingface.co/recursionpharma/nesso  
**License**: Apache 2.0  
**Paper**: https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf

## Summary

Nesso-1 is a coarse-grained cofolding model for binding affinity prediction.
Input: protein sequence (from PDB) + ligand SMILES → predicted log10(IC50/μM).

**Important: leakage.** Nesso-1 was trained on PDBbind + BindingDB + ChEMBL affinity data
and its structural trunk on PDB structures before 2021-09-30. Our test set is PDBbind v2020,
so essentially all test complexes are in Nesso's training data. The metrics below are
**in-distribution / leaked** by design — this is a ceiling reference, not a clean baseline.

## Conda Environment

```bash
conda create -y -n nesso python=3.11
/home/shpark/.conda/envs/nesso/bin/pip install "git+https://github.com/recursionpharma/nesso.git"
# Fix torch version for driver compatibility (CUDA 12.2 driver, needs cu121 build):
/home/shpark/.conda/envs/nesso/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
/home/shpark/.conda/envs/nesso/bin/pip install "huggingface_hub>=1.5.0" "transformers>=5.0.0"
/home/shpark/.conda/envs/nesso/bin/pip install pandas pyyaml
```

Nesso version: 1.0.0  
Model weights: HuggingFace `recursionpharma/nesso` revision `v1.0.0`  
(SHA: `1896c84c7186c506c7efd79051480809d51098bf`)

## Input Format

Each complex gets a YAML file in `_edrscc/yamls/{pid}.yaml`:

```yaml
sequences:
  - protein:
      id: A
      sequence: MVTPEG...  # per-chain AA sequence from {pid}_protein.pdb
  - ligand:
      id: L
      smiles: "..."
properties:
  - affinity:
      binder: L
```

All chains from `{pid}_protein.pdb` are included. If the protein has >5 chains,
only chains appearing in `{pid}_pocket.pdb` are included (to avoid memory issues).

Sequences are extracted from ATOM records using standard AA3TO1 mapping.
SMILES come from LP_PDBBind.csv `smiles` column.

## Sign Convention

Nesso outputs `affinity_pred_value` = log10(IC50/μM).  
- Lower = stronger binding (e.g., IC50 = 1 nM → log10(0.001) = -3)  
- Our pK labels = -log10(Kd or Ki in M); higher = stronger binding  

These are **negatively correlated** by convention.  
We **negate** the Nesso output before computing metrics, so that the sign
aligns with our pK labels (positive r reported):  
`pred = -log10(IC50/μM)`. The `pred` column in predictions_v2.csv is already negated.

### RMSE / pIC50 scale
`pred` sits ~6 decades below the pK scale (μM vs M units), so raw RMSE(pred, pK) ≈ 6.7 is a
pure unit offset, not a real error. We report RMSE on the **pIC50** scale:
`pIC50 = pred + 6` (μM→M decade: pIC50 = -log10(IC50/M) = -log10(IC50/μM) + 6). Pearson and
Spearman are offset-invariant, so the +6 leaves them unchanged; only RMSE becomes meaningful
(drops to ~1.4). A residual systematic offset is still expected because Nesso predicts
IC50-like potency while our labels are Kd/Ki. predictions_v2.csv has both `pred` (raw negated)
and `pred_pIC50` (= pred + 6).

## Running

### Build YAMLs

```bash
cd /home/shpark/prj-denovo/VoxBind/base/nesso
/home/shpark/.conda/envs/nesso/bin/python build_inputs.py \
    --split lp_edrscc_v2 --out_dir _edrscc/yamls --also_casf
```

### Predict (4 shards, GPUs 4-7)

```bash
# Create shard dirs first (see build_inputs.py → shard_N_yamls/)
for i in 0 1 2 3; do
    GPU=$((i+4))
    nohup bash run_shard_batch.sh $i $GPU \
        > _edrscc/logs/nohup_shard${i}.out 2>&1 &
done
```

### Score

```bash
/home/shpark/.conda/envs/nesso/bin/python score.py --out_dir _edrscc/outputs
```

## Output Files

- `_edrscc/predictions_v2.csv` — per-pid (pid, pred, pred_pIC50, pK) for v2 test set; `pred` = -log10(IC50/μM) negated, `pred_pIC50` = pred + 6 (pK-comparable scale)
- `_edrscc/results_Nesso_lp_edrscc_v2.json` — main result
- `_edrscc/results_Nesso_lp_edrscc_v2_cl{1,12,123}.json` — CL-tier results
- `base/_casf/Nesso.json` — CASF leaky/nontrain metrics

## Skipped pids

9 pids had no SMILES in LP_PDBBind.csv (all CASF-only pids):
3kck, 4ksy, 4loh, 4nxv, 5aqk, 5tzg, 6apt, 6aqo, 6hgs
