"""dump_cdg_v23_casf_preds.py — dump per-complex CASF-2016 predictions for CDG v2 & CDG v3.

gen_cdg_mse.py computes the 5-seed CASF cohort aggregates for these two ours-variants but never
saved the per-complex predictions, so they had no bootstrap CI. This reruns the IDENTICAL MSE probe
(same feature bundles, train_predict, v2 split) and writes seed CSVs in the pid,pred,y format that
base/_casf/bootstrap_casf_ci.py consumes. Non-destructive: does NOT touch casf_table1c_ours_5seed.json.

  python voxbind/test/dump_cdg_v23_casf_preds.py            # seeds 0..4
Writes base/_casf/CDG_v{2,3}_casf2016_preds_seed{s}.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import FD, REPO, build_loss, casf_eval, load_pK, train_predict, v2_split
import torch; torch.set_num_threads(2)

LOSS = build_loss("mse", 0.0)          # MSE only — matches gen_cdg_mse.py / _OURS_1C CDG v2/v3
SEEDS = range(5)
OUT = f"{REPO}/base/_casf"
ENC = {  # scorer key -> feature bundle (verbatim from gen_cdg_mse.py ENC)
    "CDG_v2": "atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt",
    "CDG_v3": "atomblob_density_gradmag_e20_v5_260823_cdg_100m_v2_interface_curriculum_0609.pt",
}


def load_feats(bn):
    import numpy as np
    d = torch.load(os.path.join(FD, bn), weights_only=False)
    f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in f.items()}


def main():
    pK, v2 = load_pK(), v2_split()
    cp, _ = casf_eval()                 # cp = all evaluable CASF pids (leaky 214)
    for key, bn in ENC.items():
        feats = load_feats(bn)
        for s in SEEDS:
            te, y, p = train_predict(feats, pK, v2, cp, s, loss_fn=LOSS)
            dst = f"{OUT}/{key}_casf2016_preds_seed{s}.csv"
            with open(dst, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["pid", "pred", "y"])
                w.writerows((q, float(pr), float(yy)) for q, pr, yy in zip(te, p, y))
            if s == 0:
                from scipy.stats import pearsonr
                import numpy as np
                r = pearsonr(np.asarray(p), np.asarray(y))[0]
                print(f"[{key}] seed0 leaky r={r:.3f}  n={len(te)} -> {os.path.basename(dst)}", flush=True)


if __name__ == "__main__":
    main()
