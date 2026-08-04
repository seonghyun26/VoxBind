"""make_cl_feats.py — derive CL-variant DSMBind feature files by subsetting v2.

The lp_edrscc_v2_cl{1,12,123} splits are exact subsets of lp_edrscc_v2 with identical
per-pid split assignments, so the frozen DSMBind encoder features (513-d, in
features/lp_edrscc_v2.pt) are byte-identical — we just keep the CL pids and copy the
split label. Writes features/{cl_split}.pt so probe_mlp.py --split <cl_split> runs
the same retrain-head protocol on the cleaned train/val/test buckets.

    <dsmbind_py> src/make_cl_feats.py
"""
import os
import csv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DSM = os.path.dirname(HERE)                                        # base/dsmbind/_edrscc
REPO = os.path.dirname(os.path.dirname(os.path.dirname(DSM)))      # VoxBind
SPLITS = os.path.join(REPO, "voxbind", "splits")
FEATS = os.path.join(DSM, "features")
VARIANTS = ["lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123"]


def main():
    base = torch.load(os.path.join(FEATS, "lp_edrscc_v2.pt"))
    for split in VARIANTS:
        clmap = {r["pid"].lower(): r["split"].lower()
                 for r in csv.DictReader(open(os.path.join(SPLITS, f"{split}.csv")))}
        feat, pK, sp = {}, {}, {}
        missing = 0
        for pid, s in clmap.items():
            if pid in base["feat"]:
                feat[pid] = base["feat"][pid]
                pK[pid] = base["pK"][pid]
                sp[pid] = s                                        # == base split (CL ⊆ v2)
            else:
                missing += 1
        out = {"feat": feat, "pK": pK, "split": sp,
               "hidden_size": base.get("hidden_size"), "dim": base["dim"]}
        torch.save(out, os.path.join(FEATS, f"{split}.pt"))
        from collections import Counter
        print(f"{split:22s} n={len(feat):5d} {dict(Counter(sp.values()))}  "
              f"missing(not-in-v2-feats)={missing}")


if __name__ == "__main__":
    main()
