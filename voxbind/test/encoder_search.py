"""encoder_search.py — rank frozen pre-trained encoders by affinity-probe metrics on the
CL3 novelty axis + LP-PDBBind.

For one cached feature bundle, trains the standard 2-layer MLP head (probe_casf_100m_mask075
machinery) on each split's TRAIN, early-stops on split-VAL, predicts split-TEST, and reports
Pearson r / Spearman rho / RMSE (mean +/- std over seeds) on four test sets:
    lp_edrscc_v2   (test 1320)
    cl123          (test 733)
    cl123_novel60  (cl123 test masked to <60% seq-id to CL3 train, 454)
    cl123_novel30  (cl123 test masked to <30%, 262)
novel60/30 are masks over the SAME cl123-test predictions (one train_predict per seed).

Out: base/_casf/encoder_search/<tag>.json   (one file per bundle → rank_encoders.py aggregates)

Usage (voxbind/):
    CUDA_VISIBLE_DEVICES=0 python test/encoder_search.py \
        --feat dataset/data/pdbbind/features/atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt \
        --seeds 5
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (  # noqa: E402
    REPO, build_loss, load_feats, load_pK, metrics, train_predict,
)

import torch  # noqa: E402
torch.set_num_threads(2)

OUT = f"{REPO}/base/_casf/encoder_search"
CL123_DIR = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"


def split_map(name):
    """pid -> {train,val,test} from a frozen split CSV."""
    m = {}
    for r in csv.DictReader(open(f"{REPO}/voxbind/splits/{name}.csv")):
        m[r["pid"].lower()] = r["split"]
    return m


def test_pids(m):
    return [p for p, s in m.items() if s == "test"]


def load_ids(path):
    return {l.strip().lower() for l in open(path) if l.strip()}


def agg(lst):
    out = {}
    for k in ("pearson", "spearman", "rmse"):
        v = [d[k] for d in lst]
        out[k] = [float(np.mean(v)), float(np.std(v))]
    out["n"] = lst[0]["n"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.feat).replace(".pt", "")
    out_path = args.out or f"{OUT}/{tag}.json"
    os.makedirs(OUT, exist_ok=True)

    feats = load_feats(args.feat)
    dim = int(next(iter(feats.values())).shape[0])
    pK = load_pK()
    loss_fn = build_loss(args.loss, 5.0)

    v2 = split_map("lp_edrscc_v2")
    cl123 = split_map("lp_edrscc_v2_cl123")
    v2_test = test_pids(v2)
    cl123_test = test_pids(cl123)
    novel60 = load_ids(f"{CL123_DIR}/cl123_test_novel60.txt")
    novel30 = load_ids(f"{CL123_DIR}/cl123_test_novel30.txt")

    per = {k: [] for k in ("lp_edrscc_v2", "cl123", "cl123_novel60", "cl123_novel30")}
    for s in range(args.seeds):
        # LP-PDBBind v2
        te, yte, pte = train_predict(feats, pK, v2, v2_test, s, loss_fn=loss_fn)
        per["lp_edrscc_v2"].append(metrics(yte, pte))
        # CL3 (train/val/test = cl123) then mask novelty over the same predictions
        te, yte, pte = train_predict(feats, pK, cl123, cl123_test, s, loss_fn=loss_fn)
        per["cl123"].append(metrics(yte, pte))
        m60 = np.array([p in novel60 for p in te])
        m30 = np.array([p in novel30 for p in te])
        per["cl123_novel60"].append(metrics(yte, pte, mask=m60))
        per["cl123_novel30"].append(metrics(yte, pte, mask=m30))

    res = {"tag": tag, "feat": args.feat, "dim": dim, "seeds": args.seeds,
           "loss": args.loss, **{k: agg(v) for k, v in per.items()}}
    json.dump(res, open(out_path, "w"), indent=2)
    line = "  ".join(
        f"{k}: rho={res[k]['spearman'][0]:.3f} r={res[k]['pearson'][0]:.3f} "
        f"rmse={res[k]['rmse'][0]:.3f} (n={res[k]['n']})"
        for k in ("lp_edrscc_v2", "cl123", "cl123_novel60", "cl123_novel30"))
    print(f"[{tag}] dim={dim}\n  {line}", flush=True)


if __name__ == "__main__":
    main()
