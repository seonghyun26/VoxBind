"""casf_seq_metrics.py — aggregate CASF-2016 preds for the seq+SMILES DTA baselines
(DeepDTA/GraphDTA/MolTrans/PLAPT) into the base/_casf/{key}.json format the results table reads
(leaky / nontrain / clean blocks, each pearson/spearman/rmse {mean,std}).

leaky   = all 214 CASF-2016 core (pK available)
nontrain= 214 minus complexes in lp_edrscc_v2 train (in_v2train flag)
clean   = casf2016_clean.csv (truly held out, not in v2 train OR val)
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
EVAL = f"{REPO}/voxbind/splits/casf2016_eval.csv"
CLEAN = f"{REPO}/voxbind/splits/casf2016_clean.csv"
CASF = f"{REPO}/base/_casf"

rows = list(csv.DictReader(open(EVAL)))
pK = {r["pid"].lower(): float(r["pK"]) for r in rows}
in_train = {r["pid"].lower(): str(r.get("in_v2train", "")).strip() in ("1", "True", "true") for r in rows}
clean_ids = {r["pid"].lower() for r in csv.DictReader(open(CLEAN))}
SUBSETS = {"leaky": lambda p: True,
           "nontrain": lambda p: not in_train.get(p, False),
           "clean": lambda p: p in clean_ids}


def metrics(pred):
    out = {}
    for name, keep in SUBSETS.items():
        pids = [p for p in pred if p in pK and keep(p)]
        if len(pids) < 3:
            continue
        y = np.array([pK[p] for p in pids]); pc = np.array([pred[p] for p in pids])
        out[name] = dict(r=float(pearsonr(pc, y)[0]), rho=float(spearmanr(pc, y).statistic),
                         rmse=float(np.sqrt(((pc - y) ** 2).mean())), n=len(pids))
    return out


def agg(seed_metrics):
    """seed_metrics: list of {subset: {r,rho,rmse,n}} → {subset: {pearson/spearman/rmse:{mean,std}, n}}."""
    out = {}
    subsets = set().union(*[set(m) for m in seed_metrics])
    for sub in subsets:
        vals = [m[sub] for m in seed_metrics if sub in m]
        if not vals:
            continue
        out[sub] = {
            "pearson":  {"mean": float(np.mean([v["r"] for v in vals])),  "std": float(np.std([v["r"] for v in vals]))},
            "spearman": {"mean": float(np.mean([v["rho"] for v in vals])), "std": float(np.std([v["rho"] for v in vals]))},
            "rmse":     {"mean": float(np.mean([v["rmse"] for v in vals])), "std": float(np.std([v["rmse"] for v in vals]))},
            "n": vals[0]["n"],
        }
    return out


def main():
    specs = {"DeepDTA": f"{CASF}/DeepDTA_casf2016_preds_seed*.csv",
             "MolTrans": f"{CASF}/MolTrans_casf2016_preds_seed*.csv",
             "PLAPT": f"{CASF}/PLAPT_casf2016_preds.csv"}
    for name, pat in specs.items():
        files = sorted(glob.glob(pat))
        if not files:
            print(f"{name}: no preds yet"); continue
        seed_metrics = []
        for f in files:
            pred = {r["pid"].lower(): float(r["pred"]) for r in csv.DictReader(open(f))}
            seed_metrics.append(metrics(pred))
        out = agg(seed_metrics)
        out["model"] = name; out["train"] = "lp_edrscc_v2 train" if name != "PLAPT" else "BindingDB (pretrained)"
        out["seeds"] = len(files)
        json.dump(out, open(f"{CASF}/{name}.json", "w"), indent=2)
        c = out.get("clean", out.get("nontrain", {}))
        print(f"{name}: clean ρ={c.get('spearman',{}).get('mean',float('nan')):.3f} n={c.get('n','?')} -> {name}.json")


if __name__ == "__main__":
    main()
