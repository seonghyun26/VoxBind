"""dump_ours_5seed_casf_preds.py — regenerate 5-seed (0..4) per-complex CASF preds for C / C+D+G / +corr.

These three had only 3 seed CSVs on disk; the multi-seed bootstrap CI (Option B) needs all 5.
Reuses the identical probe machinery (probe_casf_100m_mask075.train_predict) that produced the
originals, just over seeds 0..4. Writes base/_casf/{model}{tag}_casf2016_preds_seed{s}.csv.
Non-destructive to the JSONs.

  python voxbind/test/dump_ours_5seed_casf_preds.py     # CPU-ok (force via CUDA_VISIBLE_DEVICES="")
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (
    FD, OUT, build_loss, casf_eval, load_feats, load_pK, train_predict, v2_split)
import torch; torch.set_num_threads(2)

SEEDS = range(5)
CHAMP = f"{FD}/atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt"
COORDS = f"{FD}/atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt"
# output model name -> (feature bundle, loss_fn)
JOBS = {
    "CDG_100m_mask075":       (CHAMP,  build_loss("mse", 0.0)),        # C+D+G
    "CDG_100m_mask075_corr5": (CHAMP,  build_loss("mse+corr", 5.0)),   # C+D+G +corr
    "C_100m_mask075_coords":  (COORDS, build_loss("mse", 0.0)),        # C
}


def main():
    pK, v2 = load_pK(), v2_split()
    cp, _ = casf_eval()
    for model, (feat_path, loss_fn) in JOBS.items():
        feats = load_feats(feat_path)
        for s in SEEDS:
            te, y, p = train_predict(feats, pK, v2, cp, s, loss_fn=loss_fn)
            dst = f"{OUT}/{model}_casf2016_preds_seed{s}.csv"
            with open(dst, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["pid", "pred", "y"])
                w.writerows((q, float(pr), float(yy)) for q, pr, yy in zip(te, p, y))
        from scipy.stats import pearsonr
        import numpy as np
        r = pearsonr(np.asarray(p), np.asarray(y))[0]
        print(f"[{model}] {len(list(SEEDS))} seeds dumped; seed{s} leaky r={r:.3f}", flush=True)


if __name__ == "__main__":
    main()
