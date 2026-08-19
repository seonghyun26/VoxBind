"""casf_table1c_ours.py — 5-seed CASF re-probe of the canonical VoxBind rows for Table 1c.

Produces champion C+D+G and matched coords-only C, each at 5 seeds (MSE head), on ALL five
CASF-2016 cohorts used in results.html Table 1c:
    leaky (214) | nontrain (124) | clean-92 (92) | id<60 (64) | id<30 (32)
Frozen encoder features → MLP head trained on lp_edrscc_v2 TRAIN, early-stopped on v2-val,
predicting the 214 CASF complexes; each cohort is a mask over those predictions.

Out: base/_casf/casf_table1c_ours_5seed.json = {label: {cohort: {r/rho/rmse:[mean,std], n}}}

Usage:  cd voxbind && CUDA_VISIBLE_DEVICES=0 python test/casf_table1c_ours.py --seeds 5
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
    FD, OUT, REPO, build_loss, casf_eval, load_pK, metrics, train_predict, v2_split,
)

import torch  # noqa: E402
torch.set_num_threads(4)

BUNDLES = {
    "C+D+G": "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt",
    "C":     "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt",
}


def load_feats_basename(basename):
    d = torch.load(os.path.join(FD, basename), weights_only=False)
    feats = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in feats.items()}


def load_ids(path):
    """pid list file (whitespace/comma separated)."""
    raw = open(path).read().replace(",", " ").split()
    return {x.strip().lower() for x in raw if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--out", default=f"{OUT}/casf_table1c_ours_5seed.json")
    args = ap.parse_args()

    pK, v2 = load_pK(), v2_split()
    casf_pids, nontrain = casf_eval()
    clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    id60 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel60.txt")
    id30 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel30.txt")
    cohorts = {"leaky": set(casf_pids), "nontrain": nontrain, "clean": clean_set,
               "id60": id60, "id30": id30}
    print("cohort sizes:", {k: len(v) for k, v in cohorts.items()}, "| seeds", args.seeds, flush=True)

    loss_fn = build_loss(args.loss, 5.0)
    results = {}
    for label, basename in BUNDLES.items():
        feats = load_feats_basename(basename)
        dim = next(iter(feats.values())).shape[0]
        per = {c: [] for c in cohorts}
        for s in range(args.seeds):
            te_pids, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=loss_fn)
            for c, ids in cohorts.items():
                mask = np.array([p in ids for p in te_pids])
                if mask.sum() >= 3:
                    per[c].append(metrics(yte, pte, mask=mask))
        out = {}
        for c, lst in per.items():
            if not lst:
                continue
            out[c] = {k: [float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))]
                      for k in ("pearson", "spearman", "rmse")}
            out[c]["n"] = lst[0]["n"]
        results[label] = out
        line = "  ".join(f"{c} ρ={out[c]['spearman'][0]:.3f}(n={out[c]['n']})" for c in cohorts if c in out)
        print(f"[{label}] dim={dim}  {line}", flush=True)

    results["_meta"] = {"seeds": args.seeds, "loss": args.loss, "bundles": BUNDLES,
                        "encoder": "260705_ar_cvit_100m_v2_mask075 e49 (C+D+G); 260723 coords twin (C)"}
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
