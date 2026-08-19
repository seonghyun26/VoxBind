#!/usr/bin/env python
"""Greedy forward ensemble search over diverse CDG encoders (lp_edrscc_v2).

Ensembling is the ONLY thing that beats the champion (0.646). This searches for the best
subset by concatenating frozen features → one MLP head. Selection is by VAL ρ (honest — no
test peeking); the chosen ensemble's TEST ρ is the reported number. Known references:
best single champion ~0.646, current best ensemble (champion+v3_m090+v3_m095) ~0.656.

Run on a free gpu (capped CPU — train_one is CPU-bound):
  CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 nice -19 python test/ensemble_search.py
"""
import sys, glob, os, json
sys.path.insert(0, ".")
import numpy as np, torch
from ablation_probe import build, train_one, load_pK, split_map, FDIR

torch.set_num_threads(2)
SEARCH_SEEDS, FINAL_SEEDS = range(3), range(5)   # 3 seeds to search, 5 to confirm the pick

POOL = {                                          # name → run (features auto-resolved by glob)
    "champion_v2":       "260705_ar_cvit_100m_v2_mask075",      # 0.646, the anchor
    "v3_m085":           "260725_ar_cvit_100m_v3_m085",         # v3 data + high mask
    "v3_m090":           "260725_ar_cvit_100m_v3_m090",
    "v3_m095":           "260725_ar_cvit_100m_v3_m095",
    "v22_varmask":       "260813_cdg_100m_v22_varmask6090",     # 0.637, v2.2 + variable mask
    "v22_g7411_varmask": "260813_cdg_100m_v22_g7411_varmask7090",# 0.633, channel-sep + variable
    "depth24":           "260703_ar_cvit_g742_v2_depth24",      # deeper arch
    "big768":            "260701_ar_cvit_big768_plinder_v2",    # wider arch
    "pergroup_g7411":    "260813_cdg_100m_v22_g7411_pergroup",  # NEW pretext (may not be ready yet)
}


def resolve(run):
    hits = sorted(glob.glob(f"{FDIR}/atomblob_density_gradmag_e*_v5_{run}.pt"))
    return os.path.basename(hits[-1]) if hits else None


def evalset(members, seeds):
    data, _ = build([pool[m] for m in members], pK, sm)
    rs = [train_one(data, "mse", s) for s in seeds]
    return {k: float(np.mean([r[k] for r in rs])) for k in ("val_rho", "test_rho", "test_r", "test_rmse")}


pool = {}
for k, run in POOL.items():
    fn = resolve(run)
    if fn:
        pool[k] = fn
    else:
        print(f"  skip {k} — no features yet ({run})", flush=True)

pK, sm = load_pK(), split_map()

print("\n=== singles (3 seeds) ===", flush=True)
singles = {}
for m in sorted(pool):
    r = evalset([m], SEARCH_SEEDS)
    singles[m] = r
    print(f"  {m:<20} val {r['val_rho']:.4f}  test {r['test_rho']:.4f}", flush=True)

print("\n=== greedy forward selection (maximize VAL ρ) ===", flush=True)
selected, best_val, remaining, traj = [], -1.0, list(pool), []
while remaining:
    scored = sorted(((evalset(selected + [c], SEARCH_SEEDS), c) for c in remaining),
                    key=lambda t: t[0]["val_rho"], reverse=True)
    r, c = scored[0]
    if r["val_rho"] <= best_val + 1e-4:
        print(f"  no val gain over {best_val:.4f} → stop", flush=True)
        break
    selected.append(c); best_val = r["val_rho"]; remaining.remove(c)
    traj.append({"add": c, "members": list(selected), **r})
    print(f"  + {c:<20} → [{'+'.join(selected)}]  val {r['val_rho']:.4f}  test {r['test_rho']:.4f}", flush=True)

fin = evalset(selected, FINAL_SEEDS)
print(f"\n=== FINAL ensemble (5 seeds): {'+'.join(selected)} ===", flush=True)
print(f"  val ρ {fin['val_rho']:.4f}  test ρ {fin['test_rho']:.4f}  "
      f"test r {fin['test_r']:.4f}  rmse {fin['test_rmse']:.4f}", flush=True)
print(f"  ref: best single champion ~0.646 · known best ensemble ~0.656", flush=True)

json.dump({"selected": selected, "final": fin, "trajectory": traj,
           "singles": singles, "pool": list(pool)},
          open("../notebook/html/ensemble_search.json", "w"), indent=1)
print("  wrote notebook/html/ensemble_search.json", flush=True)
