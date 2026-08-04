# DSMBind baseline (zero-shot, pretrained) — lp_edrscc_v2

Apply the pretrained **DSMBind** drug (protein-ligand) energy model zero-shot to our
`lp_edrscc_v2` **test** split. DSMBind (Jin et al., **NeurIPS 2023**,
[arXiv 2301.10814](https://arxiv.org/abs/2301.10814);
[repo](https://github.com/wengong-jin/DSMBind)) is an **unsupervised** SE(3)
denoising-score-matching energy model — trained on crystal structures *without* affinity
labels; its predicted binding energy is meant to correlate with affinity.

No training: we load `ckpts/model.drug.allatom` and run inference (`_edrscc/src/run_eval.py`).

## Method

Follows the repo's recipe (`DrugAllAtomEnergyModel.virtual_screen`):
`parsePDB(pocket) → get_seq_coords_and_angles → (target_seq, target_coords[n_res,14,3])`;
ligand from SDF/mol2 → `binder_mol`; `DrugDataset.process` → 50-NN pocket patch;
**ESM-2-3B** (`esm2_t36_3B_UR50D`) target embedding; `model.predict` → scalar energy.

**Adaptation:** we feed the 10 Å pocket (`{pid}_pocket.pdb`) as the target, not the full
protein. The model is pocket-based (`patch_size=50`); our proteins reach 3322 residues,
which would OOM ESM-2-3B (`load_esm_embedding` does not truncate). Pocket input keeps the
binding site and bounds the ESM cost.

## Result — `lp_edrscc_v2` test (1320 complexes, 0 failures)

| metric | value |
|---|---|
| Spearman ρ (energy vs pK) | **0.363** |
| Pearson r (energy vs pK) | 0.308 |

**Heavy caveat — mostly a size artifact.** Energy correlates *positively* with pK (sign
opposite the paper's CASF `−energy` convention), and a confound analysis shows the ranking
is almost entirely molecular size:

| | ρ |
|---|---|
| ligand atom-count *alone* vs pK | 0.364 |
| energy vs ligand atom-count | 0.679 |
| **partial** ρ(energy, pK \| atom-count) | **0.168** |

DSMBind's unnormalized sum-of-pairwise-interactions energy scales with system size, and
ligand size alone predicts affinity on this diverse split just as well (ρ 0.364); only
ρ 0.168 of size-independent binding signal remains. Far below the paper's curated-CASF
numbers — expected, since our split spans ~859 diverse targets and the energy is
size-confounded. Report as a **zero-shot, size-confounded** reference, not a trained row.

## Reproduce

```bash
PY=/home/shpark/.conda/envs/dsmbind/bin/python
CUDA_VISIBLE_DEVICES=<gpu> $PY _edrscc/src/run_eval.py --gpu 0
#  -> results/dsmbind_edrscc.json + dsmbind_edrscc_preds.csv
```

## Env notes (`dsmbind` conda env)

torch 1.13.1+cu117, **SRU 3.0.0.dev6** (`pip install sru==3.0.0.dev6` — PyPI default `sru`
2.6.0 lacks `SRUpp`), fair-esm, chemprop, sidechainnet≤0.7.6, prody, rdkit, biotite. The
repo ships no top-level `bindenergy/__init__.py` or `setup.py`; we added a one-line
`bindenergy/__init__.py` (`from .models/.data/.utils import *`) and import via `sys.path`.
First run downloads ESM-2-3B (~11 GB) to the torch hub cache.
