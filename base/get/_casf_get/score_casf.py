"""score_casf.py — aggregate 3-seed predictions and write base/_casf/{KEY}.json

Usage:
  python score_casf.py \
    --model GET \
    --preds preds_GET_casf_seed0.jsonl preds_GET_casf_seed1.jsonl preds_GET_casf_seed2.jsonl \
    --manifest datasets/casf2016/manifest.csv \
    --out_json /path/to/base/_casf/GET.json
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_preds(path):
    """Return dict pid -> pred."""
    d = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            d[obj["id"]] = (float(obj["pred"]), float(obj["gt"]))
    return d


def metrics(preds, gts):
    preds = np.array(preds)
    gts = np.array(gts)
    r = pearsonr(preds, gts)[0]
    rho = spearmanr(preds, gts)[0]
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    return r, rho, rmse


def compute_split(pred_dict, pids_subset):
    """Compute pearson/spearman/rmse for a subset of pids."""
    preds, gts = [], []
    for pid in pids_subset:
        if pid in pred_dict:
            p, g = pred_dict[pid]
            preds.append(p)
            gts.append(g)
    r, rho, rmse = metrics(preds, gts)
    return r, rho, rmse, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key e.g. GET EGNN EGNN_TD")
    ap.add_argument("--preds", nargs=3, required=True, help="3 seed jsonl files")
    ap.add_argument("--manifest", required=True, help="casf manifest csv with pid,in_v2train,pK")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--train_set", default="lp_edrscc_v2 train")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    all_pids = list(manifest["pid"].astype(str).str.lower())
    nontrain_pids = list(manifest[manifest["in_v2train"] == 0]["pid"].astype(str).str.lower())

    seed_leaky = {"pearson": [], "spearman": [], "rmse": []}
    seed_nontrain = {"pearson": [], "spearman": [], "rmse": []}

    for pred_file in args.preds:
        pred_dict = load_preds(pred_file)
        r, rho, rmse, n_leaky = compute_split(pred_dict, all_pids)
        seed_leaky["pearson"].append(r)
        seed_leaky["spearman"].append(rho)
        seed_leaky["rmse"].append(rmse)

        r2, rho2, rmse2, n_nt = compute_split(pred_dict, nontrain_pids)
        seed_nontrain["pearson"].append(r2)
        seed_nontrain["spearman"].append(rho2)
        seed_nontrain["rmse"].append(rmse2)

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    result = {
        "model": args.model,
        "train": args.train_set,
        "leaky": {
            "pearson": agg(seed_leaky["pearson"]),
            "spearman": agg(seed_leaky["spearman"]),
            "rmse": agg(seed_leaky["rmse"]),
            "n": n_leaky,
        },
        "nontrain": {
            "pearson": agg(seed_nontrain["pearson"]),
            "spearman": agg(seed_nontrain["spearman"]),
            "rmse": agg(seed_nontrain["rmse"]),
            "n": n_nt,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written -> {args.out_json}")
    print(f"  leaky  ({n_leaky}):    r={np.mean(seed_leaky['pearson']):.4f}  "
          f"rho={np.mean(seed_leaky['spearman']):.4f}  "
          f"rmse={np.mean(seed_leaky['rmse']):.4f}")
    print(f"  nontrain ({n_nt}): r={np.mean(seed_nontrain['pearson']):.4f}  "
          f"rho={np.mean(seed_nontrain['spearman']):.4f}  "
          f"rmse={np.mean(seed_nontrain['rmse']):.4f}")


if __name__ == "__main__":
    main()
