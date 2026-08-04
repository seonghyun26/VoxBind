"""agg_results.py — aggregate per-seed HBGSA results into mean±std JSON.

Reads results/hbgsa_results_<tag>.csv (one row per seed×subset) and writes
results/hbgsa_<split_label>.json with the schema:
  {"pearson":{"mean":…,"std":…}, "spearman":{…}, "rmse":{…},
   "mae":{…}, "seeds":[0,1,2], "n_test":N, "tag":…, "subset":…}

Usage:
  python agg_results.py                     # aggregates all 3 CL splits
  python agg_results.py --tags edrscc_v2_cl1_3p06m  # specific tag(s)
  python agg_results.py --subset cl1        # use CL1-nested subset instead of full
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_TAGS = [
    "edrscc_v2_cl1_3p06m",
    "edrscc_v2_cl12_3p06m",
    "edrscc_v2_cl123_3p06m",
]

# map tag → output label
TAG_TO_LABEL = {
    "edrscc_v2_cl1_3p06m":   "hbgsa_cl1",
    "edrscc_v2_cl12_3p06m":  "hbgsa_cl12",
    "edrscc_v2_cl123_3p06m": "hbgsa_cl123",
}


def agg_one(tag: str, subset: str = "full") -> dict:
    csv_path = RESULTS_DIR / f"hbgsa_results_{tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    d = df[(df["tag"] == tag) & (df["subset"] == subset)]
    if d.empty:
        raise ValueError(f"No rows for tag={tag!r} subset={subset!r} in {csv_path}")
    seeds = sorted(d["seed"].tolist())
    n_test = int(d["n"].iloc[0])
    result = {
        "tag": tag,
        "subset": subset,
        "seeds": seeds,
        "n_test": n_test,
    }
    for metric in ("pearson", "spearman", "rmse", "mae"):
        vals = d[metric].values
        result[metric] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 4),
            "per_seed": [round(float(v), 4) for v in vals],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=DEFAULT_TAGS,
                    help="one or more result tags to aggregate")
    ap.add_argument("--subset", default="full",
                    help="which subset row to use: 'full' (default) or 'cl1'")
    args = ap.parse_args()

    for tag in args.tags:
        label = TAG_TO_LABEL.get(tag, tag)
        try:
            result = agg_one(tag, args.subset)
        except FileNotFoundError as e:
            print(f"  SKIP {tag}: {e}")
            continue
        out_path = RESULTS_DIR / f"{label}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  {tag} ({args.subset}):")
        print(f"    pearson  {result['pearson']['mean']:.4f} ± {result['pearson']['std']:.4f}")
        print(f"    spearman {result['spearman']['mean']:.4f} ± {result['spearman']['std']:.4f}")
        print(f"    rmse     {result['rmse']['mean']:.3f} ± {result['rmse']['std']:.3f}")
        print(f"    n_test={result['n_test']}  seeds={result['seeds']}")
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
