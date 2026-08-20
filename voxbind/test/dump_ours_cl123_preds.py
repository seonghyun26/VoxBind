"""dump_ours_cl123_preds.py — dump per-complex cl123-test predictions for our probe rows
(C, C+D+G, C+D+G+corr) so Table-1b CL3 protein-novelty cohorts can be re-scored.

Reuses probe_casf_100m_mask075.train_predict but trains the head on the lp_edrscc_v2_cl123
train/val partition and predicts the cl123 TEST set (instead of CASF). 5 seeds, per-complex
(pid, pred, y) CSV per seed -> base/_casf/cl123_seqfilter_5seed_260818/preds_ours/.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import FD, REPO, build_loss, load_pK, train_predict  # noqa: E402

import torch  # noqa: E402
torch.set_num_threads(4)

OUT = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours"
os.makedirs(OUT, exist_ok=True)

BUNDLES = {
    "C+D+G":       ("atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt", "mse"),
    "C+D+G +corr": ("atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt", "mse+corr"),
    "C":           ("atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt", "mse"),
}


def load_feats(basename):
    d = torch.load(os.path.join(FD, basename), weights_only=False)
    feats = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in feats.items()}


def cl123_split():
    m = {}
    for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv")):
        m[r["pid"].lower()] = r["split"]
    return m


def main():
    pK = load_pK()
    split = cl123_split()
    test_pids = [p for p, s in split.items() if s == "test"]
    print(f"cl123: train={sum(s=='train' for s in split.values())} "
          f"val={sum(s=='val' for s in split.values())} test={len(test_pids)}", flush=True)

    for label, (basename, loss) in BUNDLES.items():
        feats = load_feats(basename)
        loss_fn = build_loss(loss, 5.0)
        safe = label.replace(" ", "").replace("+", "_")
        for seed in range(5):
            te_pids, yte, pte = train_predict(feats, pK, split, test_pids, seed, loss_fn=loss_fn)
            path = os.path.join(OUT, f"{safe}_cl123_seed{seed}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pid", "pred", "y"])
                w.writerows((p, float(pr), float(y)) for p, pr, y in zip(te_pids, pte, yte))
        print(f"  {label}: dumped 5 seeds -> {OUT}/{safe}_cl123_seed*.csv (n_test={len(te_pids)})",
              flush=True)


if __name__ == "__main__":
    main()
