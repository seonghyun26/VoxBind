"""aggregate_dta.py — DeepDTA/MolTrans per-seed preds -> 3-seed results json (load-format).
Reads result/preds_{disp}_{split}_seed{s}.csv (pid,pred,y) -> result/{disp}_{split}.json.
v2 is aliased to lp_edrscc_v2 (the Table column key). Idempotent; run repeatedly as seeds land.
"""
import csv, glob, json, os, re
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")

def metrics(f):
    rows = list(csv.DictReader(open(f)))
    y = np.array([float(r["y"]) for r in rows]); p = np.array([float(r["pred"]) for r in rows])
    if len(rows) < 3 or np.std(p) == 0:
        return None
    return pearsonr(p, y)[0], spearmanr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean())), len(rows)

groups = {}
for f in glob.glob(os.path.join(RES, "preds_*_seed*.csv")):
    m = re.match(r"preds_(DeepDTA|MolTrans)_(.+)_seed(\d+)\.csv", os.path.basename(f))
    if m:
        groups.setdefault((m.group(1), m.group(2)), []).append(f)

for (disp, split), files in groups.items():
    per = [metrics(f) for f in files]; per = [x for x in per if x]
    if not per:
        continue
    r_, rho_, rmse_, n = zip(*per)
    out = {"method": disp, "split": split,
           "pearson": {"mean": float(np.mean(r_)), "std": float(np.std(r_))},
           "spearman": {"mean": float(np.mean(rho_)), "std": float(np.std(rho_))},
           "rmse": {"mean": float(np.mean(rmse_)), "std": float(np.std(rmse_))},
           "n_test": n[0], "seeds": len(per)}
    for key in ({split, "lp_edrscc_v2"} if split == "v2" else {split}):
        json.dump(out, open(os.path.join(RES, f"{disp}_{key}.json"), "w"), indent=2)
    print(f"{disp} {split}: rho={out['spearman']['mean']:.3f} n={n[0]} seeds={len(per)}")
