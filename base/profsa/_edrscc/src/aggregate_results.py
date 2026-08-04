"""aggregate_results.py — parse ProFSA test metrics from training logs and write results JSON.

Usage:
    python _edrscc/src/aggregate_results.py --split lp_edrscc_v2_cl1
    python _edrscc/src/aggregate_results.py --split lp_edrscc_v2_cl12
    python _edrscc/src/aggregate_results.py --split lp_edrscc_v2_cl123

Expects logs at _edrscc/logs/{split}_seed{N}.log for N in 0,1,2.
Seed 0 uses the base log name: {split}.log (or {split}_seed0.log if it exists).
Writes to base/profsa/results/profsa_{split}.json.

The script greps for the last "test/Pearson", "test/Spearman", "test/RMSE" table rows
that appear after "Restoring states from the checkpoint" in the log.
"""
import argparse
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROFSA_ROOT = os.path.dirname(os.path.dirname(HERE))  # base/profsa/
LOG_DIR = os.path.join(PROFSA_ROOT, "_edrscc", "logs")
RESULTS_DIR = os.path.join(PROFSA_ROOT, "results")
# n_test per CL split (from LMDB build)
N_TEST_MAP = {
    "lp_edrscc_v2": 1320,
    "lp_edrscc_v2_cl1": 1166,
    "lp_edrscc_v2_cl12": 1149,
    "lp_edrscc_v2_cl123": 733,
}


def parse_log(log_path):
    """Extract test Pearson/Spearman/RMSE from the ProFSA log.

    The log ends with a lightning table like:
      │  test/Pearson  │  0.626...  │
    We take the LAST occurrence (after best-ckpt restore).
    """
    with open(log_path) as f:
        text = f.read()

    # Find the block after the final "Restoring states" line
    marker = "Restoring states from the checkpoint path at"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError(f"No checkpoint-restore marker in {log_path}")
    block = text[idx:]

    def grab(key):
        # Lightning test table uses UNICODE box bars (│, U+2502), not ASCII '|' — match both.
        # e.g. "│       test/Pearson        │    0.6429203748703003     │"
        m = re.search(rf"{re.escape(key)}\s*[\|│]\s*([0-9.]+)", block)
        if m:
            return float(m.group(1))
        # fallback: the criterion INFO line "- RMSE: 1.50..." (exact-case metric name)
        name = key.split("/")[-1]                       # RMSE / Pearson / Spearman (keep case)
        m2 = re.search(rf"-\s*{re.escape(name)}:\s*([0-9.]+)", block)
        if m2:
            return float(m2.group(1))
        raise ValueError(f"Could not find '{key}' in {log_path}")

    return {
        "pearson": grab("test/Pearson"),
        "spearman": grab("test/Spearman"),
        "rmse": grab("test/RMSE"),
    }


def candidate_logs(split, seed):
    """Return candidate log paths for this split/seed (first existing wins)."""
    base = os.path.join(LOG_DIR, split)
    if seed == 0:
        return [
            f"{base}.log",
            f"{base}_seed0.log",
        ]
    return [f"{base}_seed{seed}.log"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. lp_edrscc_v2_cl1")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = {}
    missing = []

    for seed in seeds:
        cands = candidate_logs(args.split, seed)
        log_path = next((p for p in cands if os.path.exists(p)), None)
        if log_path is None:
            print(f"  [WARN] No log for seed {seed}: tried {cands}")
            missing.append(seed)
            continue
        try:
            metrics = parse_log(log_path)
            per_seed[str(seed)] = metrics
            print(f"  seed {seed}: pearson={metrics['pearson']:.4f}  spearman={metrics['spearman']:.4f}  rmse={metrics['rmse']:.4f}")
        except Exception as e:
            print(f"  [WARN] seed {seed} parse error: {e}")
            missing.append(seed)

    if not per_seed:
        print("ERROR: no seeds parsed. Run training first.")
        return

    def mean_std(key):
        vals = [v[key] for v in per_seed.values()]
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
        return round(m, 4), round(s, 4)

    p_m, p_s = mean_std("pearson")
    sp_m, sp_s = mean_std("spearman")
    r_m, r_s = mean_std("rmse")

    n_test = N_TEST_MAP.get(args.split, -1)
    result = {
        "model": "ProFSA (ICLR 2024)",
        "approach": "pretrained pocket encoder (frozen) + regression probe trained on split",
        "split": args.split,
        "subset": "test",
        "n_test": n_test,
        "seeds": list(per_seed.keys()),
        "per_seed": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in per_seed.items()},
        "mean_std": {
            "test_pearson": [p_m, p_s],
            "test_spearman": [sp_m, sp_s],
            "test_rmse": [r_m, r_s],
        },
        "config": "experiment=lba30, drugclip_reg, pretrained=profsa last.ckpt (frozen), Uni-Mol pocket+mol encoders, dropout 0.5, lr 2e-4, warmup 200, 50 epochs, best-by-val-RMSE, 1 GPU",
        "note": "ProFSA self-supervised pocket pretraining (fragment-surroundings alignment); probe head trained on CL-filtered train/val, tested on CL-filtered test.",
    }
    if missing:
        result["missing_seeds"] = missing

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"profsa_{args.split}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  pearson  {p_m:.4f} ± {p_s:.4f}")
    print(f"  spearman {sp_m:.4f} ± {sp_s:.4f}")
    print(f"  rmse     {r_m:.4f} ± {r_s:.4f}")


if __name__ == "__main__":
    main()
