"""casf_5seed.py — 5-seed CASF-2016 probe over the best CDG encoders (+ ensembles).

Question: with 5 seeds, does any of our good encoders reach ProFSA on the honest
CASF-2016 *clean* subset (ρ ≈ 0.676, the strongest structure baseline)?

Reuses the frozen-encoder → MLP-head probe from ``probe_casf_100m_mask075.py``:
train head on lp_edrscc_v2 TRAIN (cached epoch features), early-stop on v2-val, predict
CASF-2016; report leaky(214) / nontrain(124) / clean(92) ρ·r·RMSE (mean±std over seeds).
Ensembles = feature-concat across encoders (same as test/ensemble_robust.py), one head.

Usage:
    cd voxbind && CUDA_VISIBLE_DEVICES=6 python test/casf_5seed.py --seeds 5
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (  # noqa: E402
    FD, OUT, REPO, build_loss, casf_eval, load_pK, metrics, train_predict, v2_split,
)

torch.set_num_threads(4)

# ── candidate encoders (label -> exact feature bundle basename) ──────────────────
SINGLES = {
    "champion_v2_m075_e49":  "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt",
    "champion_v2_m075_e100": "atomblob_density_gradmag_e99_v5_260705_ar_cvit_100m_v2_mask075_e100.pt",
    "v3_m085":               "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m085.pt",
    "v3_m090":               "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m090.pt",
    "v3_m095":               "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m095.pt",
    "v2_m085":               "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v2_m085.pt",
    "v2_m090":               "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v2_m090.pt",
    "v2_ep100_e25":          "atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt",
    "epochpeak_e59":         "atomblob_density_gradmag_e59_v5_260708_ar_cvit_100m_v2_epochpeak_e59.pt",
    "v3_m075_hcs015":        "atomblob_density_gradmag_e49_v5_260722_ar_cvit_100m_v3_m075_hcs015.pt",
    "d2vaux05":              "atomblob_density_gradmag_e49_v5_260806_cdg_100m_v2_d2vaux05.pt",
    "v22_m075":              "atomblob_density_gradmag_e49_v5_260813_cdg_100m_v22_mask075.pt",
}

# ── ensembles (feature-concat over these single labels) ─────────────────────────
ENSEMBLES = {
    "ens_champ+v3(090,095)":      ["champion_v2_m075_e49", "v3_m090", "v3_m095"],
    "ens_champ+v3(085,090,095)":  ["champion_v2_m075_e49", "v3_m085", "v3_m090", "v3_m095"],
    "ens_v3(085,090,095)":        ["v3_m085", "v3_m090", "v3_m095"],
    "ens_champ+e100+v3(090,095)": ["champion_v2_m075_e49", "champion_v2_m075_e100", "v3_m090", "v3_m095"],
}

PROFSA_CLEAN = 0.676  # strongest structure baseline on CASF-2016 clean (n=92)


def load_feats_basename(basename):
    d = torch.load(os.path.join(FD, basename), weights_only=False)
    feats = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in feats.items()}


def concat_feats(basenames):
    """Concatenate per-pid feature vectors across bundles (intersection of pids)."""
    dicts = [load_feats_basename(b) for b in basenames]
    common = set(dicts[0])
    for d in dicts[1:]:
        common &= set(d)
    return {p: np.concatenate([d[p] for d in dicts]).astype(np.float32) for p in common}


def run_condition(label, feats, pK, v2, casf_pids, nontrain, clean_set, loss_fn, seeds):
    per = {"leaky": [], "nontrain": [], "clean": []}
    for s in range(seeds):
        te_pids, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=loss_fn)
        nt = np.array([p in nontrain for p in te_pids])
        cl = np.array([p in clean_set for p in te_pids])
        per["leaky"].append(metrics(yte, pte))
        per["nontrain"].append(metrics(yte, pte, mask=nt))
        per["clean"].append(metrics(yte, pte, mask=cl))

    def agg(lst):
        out = {k: {"mean": float(np.mean([d[k] for d in lst])),
                   "std": float(np.std([d[k] for d in lst]))}
               for k in ("pearson", "spearman", "rmse")}
        out["n"] = lst[0]["n"]
        return out

    return {k: agg(v) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--losses", default="mse,mse+corr", help="comma list: mse | mse+corr")
    ap.add_argument("--aux_weight", type=float, default=5.0)
    ap.add_argument("--out", default=f"{OUT}/casf_5seed_summary.json")
    args = ap.parse_args()

    pK, v2 = load_pK(), v2_split()
    casf_pids, nontrain = casf_eval()
    clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    losses = [l.strip() for l in args.losses.split(",") if l.strip()]
    print(f"CASF: {len(casf_pids)} leaky | {len(nontrain)} nontrain | {len(clean_set)} clean | "
          f"seeds={args.seeds} | losses={losses} | ProFSA-clean ρ={PROFSA_CLEAN}\n", flush=True)

    results = {}
    conditions = [("single", k, [v]) for k, v in SINGLES.items()] + \
                 [("ensemble", k, [SINGLES[m] for m in members]) for k, members in ENSEMBLES.items()]

    for kind, label, basenames in conditions:
        feats = load_feats_basename(basenames[0]) if len(basenames) == 1 else concat_feats(basenames)
        dim = next(iter(feats.values())).shape[0]
        for loss in losses:
            loss_fn = build_loss(loss, args.aux_weight)
            tag = f"{label}::{loss}"
            res = run_condition(label, feats, pK, v2, casf_pids, nontrain, clean_set,
                                loss_fn, args.seeds)
            res.update(kind=kind, dim=dim, loss=loss, n_encoders=len(basenames))
            results[tag] = res
            c = res["clean"]["spearman"]
            print(f"  {tag:<46} dim={dim:<5} clean ρ={c['mean']:.4f}±{c['std']:.4f}  "
                  f"(leaky {res['leaky']['spearman']['mean']:.3f} / "
                  f"nontrain {res['nontrain']['spearman']['mean']:.3f})", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # ── ranked summary on the honest clean subset ────────────────────────────────
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["clean"]["spearman"]["mean"])
    print(f"\n{'='*78}\nCASF-2016 CLEAN (n=92) — ranked by Spearman ρ  [ProFSA={PROFSA_CLEAN}]\n{'='*78}")
    print(f"{'condition':<46} {'clean ρ':>14} {'clean RMSE':>12}  vs ProFSA")
    for tag, r in ranked:
        c = r["clean"]
        gap = c["spearman"]["mean"] - PROFSA_CLEAN
        flag = "≥ ProFSA" if gap >= -0.003 else f"{gap:+.3f}"
        print(f"{tag:<46} {c['spearman']['mean']:.3f}±{c['spearman']['std']:.3f} "
              f"{c['rmse']['mean']:8.3f}     {flag}")


if __name__ == "__main__":
    main()
