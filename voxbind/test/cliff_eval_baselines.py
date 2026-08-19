#!/usr/bin/env python3
"""Cliff metrics for the SUPERVISED baselines (TargetDiff, HBGSA), reusing the exact cliff
definitions + labels from cliff_eval_canonical.json (no retraining). Same metrics as canonical:
  sign_acc  = mean over test-test cliff pairs of sign(pred_a-pred_b)==sign(pK_a-pK_b)   (↑)
  rmse_cliff= RMSE on cliff mols                                                          (↓)
  roughslope= spearman(|pred-pK|, local SAR roughness) over mols with a similar analog    (↓)
  sanity    = spearman(pred, pK) over covered test mols                                   (↑)
Baselines that don't cover every pid are scored on the COVERED subset (n reported).
Writes voxbind/dataset/data/pdbbind/cliff_eval_baselines.json.
"""
import json, csv, os
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/shpark/prj-denovo/VoxBind"
CAN = f"{ROOT}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"
OUT = f"{ROOT}/voxbind/dataset/data/pdbbind/cliff_eval_baselines.json"
TD_CSV = f"{ROOT}/notebook/html/scatter/scatter_egnn_targetdiff.csv"
HB_DIR = f"{ROOT}/base/hbgsa/results"
HB_TAG = "preds_edrscc_40m"      # best test/cliff coverage on lp_edrscc_v2

d = json.load(open(CAN))
test_pids = d["test_pids"]; pK = d["pK"]; cliff = d["cliff"]
tt_pairs = cliff["tt_pairs"]                 # [[pid_a, pid_b], ...]
test_cliff = cliff["test_cliff"]; noncliff = cliff["noncliff"]; rough = cliff["roughness"]
has_nb = [p for p in test_pids if rough.get(p, 0) > 0]

def metrics(pidpred):
    """pidpred: dict pid->pred. Metrics on the covered subset."""
    cov = set(pidpred)
    tt = [(a, b) for a, b in tt_pairs if a in cov and b in cov]
    tc = [p for p in test_cliff if p in cov]
    nb = [p for p in has_nb if p in cov]
    ts = [p for p in test_pids if p in cov]
    sign = np.mean([np.sign(pidpred[a]-pidpred[b]) == np.sign(pK[a]-pK[b]) for a, b in tt]) if tt else float("nan")
    rc = np.sqrt(np.mean([(pidpred[p]-pK[p])**2 for p in tc])) if tc else float("nan")
    rn = np.sqrt(np.mean([(pidpred[p]-pK[p])**2 for p in [q for q in noncliff if q in cov]]))
    err = np.array([abs(pidpred[p]-pK[p]) for p in nb]); rg = np.array([rough[p] for p in nb])
    rs = float(spearmanr(err, rg).statistic) if len(nb) > 2 else float("nan")
    rho = float(spearmanr([pidpred[p] for p in ts], [pK[p] for p in ts]).statistic)
    return dict(sign_acc=float(sign), rmse_cliff=float(rc), rmse_noncliff=float(rn),
                roughslope=rs, rho=rho, n_tt=len(tt), n_cliff=len(tc), n_test=len(ts))

res = {"scheme": d["scheme"], "n_tt_all": len(tt_pairs), "n_test_cliff_all": len(test_cliff),
       "models": {}}

# ── TargetDiff (single prediction per pid) ──────────────────────────────
td = {r["pid"]: float(r["y_pred"]) for r in csv.DictReader(open(TD_CSV))}
td = {p: v for p, v in td.items() if p in set(test_pids)}
res["models"]["TargetDiff / EGNN"] = metrics(td)

# ── HBGSA (3 seeds → mean±std) ──────────────────────────────────────────
hb_seed_metrics = []
for s in range(3):
    f = f"{HB_DIR}/{HB_TAG}_seed{s}.json"
    if not os.path.exists(f):
        continue
    h = json.load(open(f))
    ids, ps = h["pdb_id"], h["pred"]        # each seed file has its own "pred"
    hp = {p: float(v) for p, v in zip(ids, ps) if p in set(test_pids)}
    hb_seed_metrics.append(metrics(hp))
def agg(key):
    vals = [m[key] for m in hb_seed_metrics]
    return [float(np.mean(vals)), float(np.std(vals))]
if hb_seed_metrics:
    res["models"]["HBGSA"] = {k: agg(k) for k in ["sign_acc", "rmse_cliff", "rmse_noncliff", "roughslope", "rho"]}
    res["models"]["HBGSA"].update({k: hb_seed_metrics[0][k] for k in ["n_tt", "n_cliff", "n_test"]})
    res["models"]["HBGSA"]["n_seeds"] = len(hb_seed_metrics)

json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}\n")
print(f"{'baseline':22s}{'ρ':>8}{'sign-acc':>10}{'RMSE_cliff':>12}{'roughslope':>12}   coverage")
print("-"*78)
for m, r in res["models"].items():
    def g(k):
        v = r[k]; return (v[0] if isinstance(v, list) else v)
    print(f"{m:22s}{g('rho'):>8.3f}{g('sign_acc'):>10.3f}{g('rmse_cliff'):>12.3f}"
          f"{g('roughslope'):>12.3f}   tt={r['n_tt']}/{len(tt_pairs)} cliff={r['n_cliff']}/{len(test_cliff)} test={r['n_test']}")
print("\nreference (canonical): C(coords) sign0.714 rmse1.546 rough0.168 | C+D+G sign0.635 rmse1.563 rough0.196")
