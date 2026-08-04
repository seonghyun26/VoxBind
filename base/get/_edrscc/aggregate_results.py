"""aggregate_results.py — Collect 3-seed inference results into a single results JSON.

Reads the per-seed preds_<tag>_seed{0,1,2}.jsonl files, computes per-seed metrics, and
writes base/get/_edrscc/results_{method}_{split}.json with schema:
    {
        "pearson":  {"mean": float, "std": float},
        "spearman": {"mean": float, "std": float},
        "rmse":     {"mean": float, "std": float},
        "seeds": [0, 1, 2],
        "n_test": int,
        "per_seed": {
            "0": {"pearson": float, "spearman": float, "rmse": float},
            ...
        }
    }

Usage (from base/get/):
    python _edrscc/aggregate_results.py --method GET --split cl1
    python _edrscc/aggregate_results.py --method EGNN --split v2
    python _edrscc/aggregate_results.py --method EGNN_TD --split cl123

The preds files must already exist at:
    _edrscc/preds_{method}_{split}_seed{s}.jsonl   (s in 0,1,2)
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def load_preds(path):
    rows = [json.loads(l) for l in open(path)]
    return {r["id"]: (r["label"], r["gt"]) for r in rows}


def metrics(preds_dict):
    ids = sorted(preds_dict)
    y = np.array([preds_dict[i][1] for i in ids])
    yhat = np.array([preds_dict[i][0] for i in ids])
    yhat = np.where(np.isfinite(yhat), yhat, 0.0)
    r = float(pearsonr(y, yhat)[0])
    rho = float(spearmanr(y, yhat)[0])
    rmse = float(np.sqrt(((y - yhat) ** 2).mean()))
    return {"pearson": r, "spearman": rho, "rmse": rmse, "n": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help="Method tag: GET, EGNN, EGNN_TD")
    ap.add_argument("--split", required=True,
                    help="Split tag: v2, cl1, cl12, cl123")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    per_seed = {}
    for s in args.seeds:
        fname = os.path.join(HERE, f"preds_{args.method}_{args.split}_seed{s}.jsonl")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing: {fname}")
        p = load_preds(fname)
        m = metrics(p)
        per_seed[str(s)] = m
        print(f"  seed{s}: r={m['pearson']:.4f} rho={m['spearman']:.4f} rmse={m['rmse']:.4f}  n={m['n']}")

    n_test = per_seed[str(args.seeds[0])]["n"]
    rs = [per_seed[str(s)]["pearson"] for s in args.seeds]
    rhos = [per_seed[str(s)]["spearman"] for s in args.seeds]
    rmses = [per_seed[str(s)]["rmse"] for s in args.seeds]

    result = {
        "method": args.method,
        "split": args.split,
        "pearson":  {"mean": float(np.mean(rs)),   "std": float(np.std(rs))},
        "spearman": {"mean": float(np.mean(rhos)),  "std": float(np.std(rhos))},
        "rmse":     {"mean": float(np.mean(rmses)), "std": float(np.std(rmses))},
        "seeds": list(args.seeds),
        "n_test": n_test,
        "per_seed": per_seed,
    }
    out_path = os.path.join(HERE, f"results_{args.method}_{args.split}.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n  pearson  {result['pearson']['mean']:.4f} ± {result['pearson']['std']:.4f}")
    print(f"  spearman {result['spearman']['mean']:.4f} ± {result['spearman']['std']:.4f}")
    print(f"  rmse     {result['rmse']['mean']:.4f} ± {result['rmse']['std']:.4f}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
