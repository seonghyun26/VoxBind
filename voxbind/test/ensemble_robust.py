#!/usr/bin/env python
"""Robust 5-seed evaluation of the most promising ensemble combos (lp_edrscc_v2).

The greedy search used 3 seeds and missed the known 3-member (champion+v3_m090+v3_m095)
because val ρ is noisy at 3 seeds. This evaluates explicit combos at 5 seeds and reports
the VAL-selected best (honest) alongside every combo's test ρ, so we know whether an
ensemble genuinely beats champion single (0.646).
"""
import sys, glob, os, json
sys.path.insert(0, ".")
import numpy as np, torch
from ablation_probe import build, train_one, load_pK, split_map, FDIR

torch.set_num_threads(2)
R = {
    "champion": "260705_ar_cvit_100m_v2_mask075",
    "v3_085":   "260725_ar_cvit_100m_v3_m085",
    "v3_090":   "260725_ar_cvit_100m_v3_m090",
    "v3_095":   "260725_ar_cvit_100m_v3_m095",
    "v22vm":    "260813_cdg_100m_v22_varmask6090",
    "pergroup": "260813_cdg_100m_v22_g7411_pergroup",
}
def resolve(run):
    h = sorted(glob.glob(f"{FDIR}/atomblob_density_gradmag_e*_v5_{run}.pt"))
    return os.path.basename(h[-1]) if h else None
F = {k: resolve(v) for k, v in R.items()}

COMBOS = [
    ["champion"],                                    # single ref
    ["v3_095", "v3_090"],                            # greedy pick
    ["champion", "v3_090", "v3_095"],                # known "best" (0.656 test)
    ["champion", "v3_090"],
    ["champion", "v3_095"],
    ["v3_085", "v3_090", "v3_095"],                  # all-v3
    ["champion", "v22vm", "v3_095"],                 # cross-data diverse
    ["champion", "v3_085", "v3_090", "v3_095"],      # 4-member
    ["champion", "v3_090", "v3_095", "v22vm"],       # 4-member + v2.2
    ["champion", "v3_090", "v3_095", "pergroup"],    # + new pretext
]

pK, sm = load_pK(), split_map()
rows = []
print("=== ensemble combos (5 seeds) ===", flush=True)
for c in COMBOS:
    data, _ = build([F[m] for m in c], pK, sm)
    rs = [train_one(data, "mse", s) for s in range(5)]
    agg = {k: float(np.mean([r[k] for r in rs])) for k in ("val_rho", "test_rho", "test_r", "test_rmse")}
    agg["test_rho_std"] = float(np.std([r["test_rho"] for r in rs]))
    agg["combo"] = "+".join(c)
    rows.append(agg)
    print(f"  {agg['combo']:<42} val {agg['val_rho']:.4f}  test {agg['test_rho']:.4f}±{agg['test_rho_std']:.4f}  rmse {agg['test_rmse']:.4f}", flush=True)

champ = [r for r in rows if r["combo"] == "champion"][0]["test_rho"]
best = max(rows, key=lambda r: r["val_rho"])
print(f"\n  VAL-selected best: {best['combo']}  test ρ {best['test_rho']:.4f}  (champion single {champ:.4f}, Δ{best['test_rho']-champ:+.4f})", flush=True)
json.dump(rows, open("../notebook/html/ensemble_robust.json", "w"), indent=1)
print("  wrote notebook/html/ensemble_robust.json", flush=True)
