"""dump_v2preds.py — per-pid v2-test predictions for the DSMBind MLP-probe (frozen DSMBind
features + head, 3 seeds), matching Table-1a's DSMBind row, so the LP protein-novelty subsets
can be re-scored consistently. Writes base/_casf/novel_preds/DSMBind_v2_seed{s}.csv (pid,pred,y)."""
import os, csv
import numpy as np
from probe_mlp import load_split, train_one

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.abspath(os.path.join(DSM_ROOT, "..", "_casf", "novel_preds")); os.makedirs(OUT, exist_ok=True)

def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pt = os.path.join(DSM_ROOT, "_edrscc", "features", "lp_edrscc_v2.pt")
    data, dim = load_split(pt)
    _, yte, te_pids = data["test"]
    ymap = dict(zip(te_pids, yte.tolist()))
    for s in range(3):
        m, preds = train_one(data, seed=s, dim=dim, device=dev)
        with open(f"{OUT}/DSMBind_v2_seed{s}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "pred", "y"])
            for pid, v in preds.items():
                w.writerow([pid, float(v), float(ymap[pid])])
        print(f"DSMBind seed{s}: r={m['pearson']:.3f} rho={m['spearman']:.3f} n={len(preds)}")

if __name__ == "__main__":
    main()
