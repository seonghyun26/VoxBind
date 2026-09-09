#!/usr/bin/env python3
"""Ensemble figures from ensemble_results.json.
fig1: 6 cohorts x 3 metrics (r, rho, RMSE) — champion baseline + champion⊕{GET,EGNN,CheapNet,ProFSA}.
fig2: Spearman rho, champion vs partner-ALONE vs ensemble per cohort (does the ensemble beat the partner alone?).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{HERE}/ensemble_results.json"))
COH = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
COHL = ["FULL", "CL3", "CL3\nID60", "CL3\nID30", "CASF\nnontrain", "CASF\nclean"]
PARTNERS = ["GET", "EGNN", "CheapNet", "ProFSA"]
COLORS = {"GET": "#1f77b4", "EGNN": "#2ca02c", "CheapNet": "#ff7f0e", "ProFSA": "#d62728"}
x = np.arange(len(COH))

def series(d, metric):
    return [d.get(c, {}).get(metric, np.nan) for c in COH]

# ---------- fig1: 6 cohorts x 3 metrics ----------
METRICS = [("r", "Pearson r", False), ("rho", "Spearman ρ", False), ("rmse", "RMSE", True)]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (mk, ml, lower_better) in zip(axes, METRICS):
    ax.plot(x, series(R["champion"], mk), "-o", color="black", lw=2.6, ms=7, label="champion (C+D+G)", zorder=5)
    for p in PARTNERS:
        ax.plot(x, series(R["ensembles"][p], mk), "--o", color=COLORS[p], lw=1.8, ms=5, label=f"⊕ {p}")
    ax.set_xticks(x); ax.set_xticklabels(COHL, fontsize=9)
    ax.set_title(ml + ("  (↓ better)" if lower_better else "  (↑ better)"), fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.axvspan(3.5, 5.5, color="grey", alpha=0.06)  # shade CASF holdout region
    if mk == "r":
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
fig.suptitle(f"Probe-side feature ensemble: champion ⊕ baseline encoder (PCA-64), 5-seed, n_shared={R['n_shared']}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/fig1_ensemble_6set_3metric.{ext}", dpi=150, bbox_inches="tight")
print("wrote fig1")

# ---------- fig2: rho, champion vs partner-ALONE vs ensemble ----------
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
for ax, p in zip(axes2, PARTNERS):
    w = 0.27
    ax.bar(x - w, series(R["champion"], "rho"), w, color="black", label="champion")
    ax.bar(x,     series(R["alone"][p], "rho"), w, color=COLORS[p], alpha=0.45, label=f"{p} alone")
    ax.bar(x + w, series(R["ensembles"][p], "rho"), w, color=COLORS[p], label=f"champion ⊕ {p}")
    ax.set_xticks(x); ax.set_xticklabels(COHL, fontsize=8)
    ax.set_title(p, fontsize=12, fontweight="bold"); ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0.40, 0.75); ax.legend(fontsize=8, loc="lower left")
axes2[0].set_ylabel("Spearman ρ", fontsize=11)
fig2.suptitle("Is the ensemble better than the partner ALONE?  (Spearman ρ per cohort)", fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig2.savefig(f"{HERE}/fig2_ensemble_vs_alone.{ext}", dpi=150, bbox_inches="tight")
print("wrote fig2")

# ---------- print a compact Δρ table ----------
print("\nΔρ (ensemble − champion) per cohort:")
print(f"{'cohort':<15}" + "".join(f"{p:>10}" for p in PARTNERS))
for c in COH:
    row = f"{c:<15}"
    for p in PARTNERS:
        d = R["ensembles"][p].get(c, {}).get("rho", np.nan) - R["champion"].get(c, {}).get("rho", np.nan)
        row += f"{d:>+10.3f}"
    print(row)
