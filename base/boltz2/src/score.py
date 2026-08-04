"""score.py — collect Boltz-2 affinity predictions and score against lp_edrscc_v2 pK.

Boltz-2 outputs `affinity_pred_value` = log10(IC50 in uM) (lower = stronger binder).
We convert to a pIC50-like predicted "pK":  pred_pK = 6 - affinity_pred_value
(pIC50 = -log10(IC50 in M) = 6 - log10(IC50 in uM)), then correlate with the
experimental pK label. We also score the raw binder probability as a ranking signal.

NOTE on interpretation (see README): this is an IN-DISTRIBUTION number for Boltz-2
(PDBbind almost certainly overlaps its training data) and Boltz-2 affinity is designed
for within-series hit-to-lead ranking, not cross-target absolute affinity — so treat the
correlation on this diverse 1320-target split as a heavily-caveated reference, not a clean
held-out comparison to the frozen probes.
"""
import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def find_affinity_json(preds_dir, pid):
    # boltz writes predictions/<name>/affinity_<name>.json (layout can vary by version)
    cands = glob.glob(os.path.join(preds_dir, "**", f"affinity_{pid}.json"), recursive=True)
    cands += glob.glob(os.path.join(preds_dir, "**", pid, "affinity*.json"), recursive=True)
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default=os.path.join(BASE, "preds"))
    ap.add_argument("--manifest", default=os.path.join(BASE, "results", "input_manifest.csv"))
    ap.add_argument("--tag", default="edrscc")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["pid"] = man["pid"].astype(str).str.lower()
    evaluable = man[man["status"].isin(["ok", "ok_large_ligand"])].copy()

    recs = []
    for _, r in evaluable.iterrows():
        pid = r["pid"]
        j = find_affinity_json(args.preds_dir, pid)
        if j is None:
            continue
        with open(j) as f:
            a = json.load(f)
        val = a.get("affinity_pred_value")
        prob = a.get("affinity_probability_binary")
        if val is None:
            continue
        recs.append((pid, float(r["pK"]), float(val), 6.0 - float(val),
                     float(prob) if prob is not None else np.nan))

    df = pd.DataFrame(recs, columns=["pid", "pK", "affinity_log10IC50_uM", "pred_pK", "binder_prob"])
    n_eval = len(evaluable); n_done = len(df)
    print(f"evaluable inputs: {n_eval}  | predictions found: {n_done}  "
          f"({100*n_done/max(n_eval,1):.1f}% coverage)")
    if n_done < 3:
        print("too few predictions to score"); return

    g = df["pK"].values
    p = df["pred_pK"].values
    pearson = float(np.corrcoef(g, p)[0, 1])
    spearman = float(stats.spearmanr(g, p)[0])
    rmse = float(np.sqrt(((g - p) ** 2).mean()))
    prob_sp = float(stats.spearmanr(g, df["binder_prob"].values)[0]) if df["binder_prob"].notna().any() else None

    summary = {
        "model": "Boltz-2 (pretrained affinity module, zero-shot)",
        "split": "lp_edrscc_v2", "subset": "test",
        "n_evaluable": int(n_eval), "n_predicted": int(n_done),
        "test_pearson": pearson, "test_spearman": spearman, "test_rmse": rmse,
        "binder_prob_spearman": prob_sp,
        "note": "pred_pK = 6 - log10(IC50_uM); IN-DISTRIBUTION for Boltz-2 (PDBbind overlap); "
                "affinity head designed for within-series ranking, not cross-target absolute pK.",
    }
    with open(os.path.join(BASE, "results", f"boltz2_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(os.path.join(BASE, "results", f"boltz2_{args.tag}_preds.csv"), index=False)

    print(f"\n========= Boltz-2 zero-shot  lp_edrscc_v2 (test) =========")
    print(f"n = {n_done}")
    print(f"test Pearson r  {pearson:.4f}")
    print(f"test Spearman   {spearman:.4f}")
    print(f"test RMSE       {rmse:.4f}  (pred_pK vs pK; IC50-vs-Kd/Ki unit mismatch -> read correlation, not RMSE)")
    if prob_sp is not None:
        print(f"binder-prob Spearman vs pK  {prob_sp:.4f}")
    print(f"saved -> results/boltz2_{args.tag}.json")


if __name__ == "__main__":
    main()
