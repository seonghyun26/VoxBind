#!/usr/bin/env python3
"""Activity-cliff eval on the EXTERNAL MoleculeACE benchmark filtered to density-backed
structures (voxbind/dataset/data/moleculeace_density/manifest.json). 17 expert cliff pairs
(both members co-crystallised with experimental density); each mol → its best-resolution PDB
entry → the model's affinity prediction; label = MoleculeACE expert pKi.

To stay out-of-sample we keep only pairs whose BOTH members are in the lp_edrscc_v2 TEST split
(train/val members would be in-sample for the frozen probe). All models are scored on the SAME
common-covered pair set for a fair comparison.
  sign_acc  = mean sign(pred_a-pred_b)==sign(pKi_a-pKi_b)   (↑; scale-free, the trustworthy metric)
  rmse_cliff= RMSE(pred, pKi) over the pair molecules         (↓; assay-offset caveat)
Writes voxbind/dataset/data/pdbbind/cliff_eval_moleculeace.json.
"""
import json, csv
import numpy as np

ROOT = "/home/shpark/prj-denovo/VoxBind"
MAN = f"{ROOT}/voxbind/dataset/data/moleculeace_density/manifest.json"
CAN = f"{ROOT}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"
TD_CSV = f"{ROOT}/notebook/html/260625/scatter_egnn_targetdiff.csv"
HB = [f"{ROOT}/base/hbgsa/results/preds_edrscc_40m_seed{s}.json" for s in range(3)]
OUT = f"{ROOT}/voxbind/dataset/data/pdbbind/cliff_eval_moleculeace.json"

# ── expert cliff pairs (best-entry pid + MoleculeACE pKi + split) ────────────
M = json.load(open(MAN))
pairs = []
for tgt, rec in M.items():
    mols = rec.get("density_mols", [])
    for i, j in rec.get("pairs", []):
        a, b = mols[i], mols[j]
        pairs.append((tgt, a["best"], a["pKi"], a["split"], b["best"], b["pKi"], b["split"]))
# out-of-sample = BOTH members' best-entry in OUR lp_edrscc_v2 TEST set (the frozen probe only
# predicts test pids, so this keeps every model on out-of-sample complexes). The manifest "split"
# is MoleculeACE's own split — irrelevant here.
can = json.load(open(CAN))
CAN_TEST = set(can["test_pids"])
oos = [p for p in pairs if p[1] in CAN_TEST and p[4] in CAN_TEST]

# ── model predictions (pid -> pred), seed-averaged where available ───────────
def avg(pred_dict):  # {pid:[seed...]} -> {pid:mean}
    return {p: float(np.mean(v)) for p, v in pred_dict.items()}
preds = {
    "C (ViT)":            avg(can["preds"]["C (ViT)"]),
    "C+D+G (ChannelViT)": avg(can["preds"]["C+D+G (ChannelViT)"]),
    "TargetDiff / EGNN":  {r["pid"]: float(r["y_pred"]) for r in csv.DictReader(open(TD_CSV))},
}
hb_seed = []
for f in HB:
    h = json.load(open(f)); hb_seed.append({p: float(v) for p, v in zip(h["pdb_id"], h["pred"])})
preds["HBGSA"] = {p: float(np.mean([hs[p] for hs in hb_seed if p in hs]))
                  for p in set().union(*[set(hs) for hs in hb_seed])}

# ── common covered pair set (both members predicted by EVERY model) ──────────
def covered(model, plist):
    d = preds[model]; return [p for p in plist if p[1] in d and p[4] in d]
common = [p for p in oos if all(p[1] in preds[m] and p[4] in preds[m] for m in preds)]
print(f"MoleculeACE density cliff pairs: {len(pairs)} total | {len(oos)} both-in-test | "
      f"{len(common)} covered by ALL models")

def sanity_rho(model):
    """bulk Spearman on the full lp_edrscc_v2 test set (reference)."""
    from scipy.stats import spearmanr
    d = preds[model]; pK = can["pK"]; ts = [p for p in can["test_pids"] if p in d]
    return float(spearmanr([d[p] for p in ts], [pK[p] for p in ts]).statistic)

res = {"n_pairs_total": len(pairs), "n_both_test": len(oos), "n_common": len(common),
       "targets": sorted(set(p[0] for p in common)), "models": {}}
for m in preds:
    sa = np.mean([np.sign(preds[m][p[1]]-preds[m][p[4]]) == np.sign(p[2]-p[5]) for p in common])
    mols = {}
    for p in common:
        mols[p[1]] = p[2]; mols[p[4]] = p[5]
    rc = np.sqrt(np.mean([(preds[m][pid]-pki)**2 for pid, pki in mols.items()]))
    res["models"][m] = {"sign_acc": float(sa), "rmse_cliff": float(rc),
                        "n_pairs": len(common), "n_mols": len(mols), "sanity_rho": sanity_rho(m)}

json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}\n")
print(f"{'model':22s}{'sanity ρ':>10}{'sign-acc':>10}{'RMSE_cliff':>12}   (n={len(common)} pairs)")
print("-"*64)
for m, r in res["models"].items():
    print(f"{m:22s}{r['sanity_rho']:>10.3f}{r['sign_acc']:>10.3f}{r['rmse_cliff']:>12.3f}")
print(f"\ntargets in common set: {res['targets']}")
