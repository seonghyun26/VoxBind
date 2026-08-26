#!/usr/bin/env python
"""5-metric panel for an encoder checkpoint: CL3 / CL3-ID60 / CL3-ID30 / CASF / CASF-clean.
All Spearman ρ (mean±std over seeds). Reads:
  - CL3 per-pid preds  : results/scatter_panel_cl3/scatter_e{EP}_v5_lp_edrscc_v2_cl123_{RUN}_cl3dump_seed*.csv
  - CL3 novel cohorts  : base/_casf/cl123_seqfilter_5seed_260818/cl123_test_novel{60,30}.txt
  - CASF per-pid preds : base/_casf/{RUN}_e{EP}_corr5_casf2016_preds_seed*.csv  (leaky=214, clean=92)
Usage:  python scripts/panel5.py <RUN> <EP> [label]
"""
import csv, glob, sys, os, statistics as st
import numpy as np
from scipy.stats import spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
VOX = f"{REPO}/voxbind"
CL3_DIR = f"{VOX}/dataset/data/pdbbind/results/scatter_panel_cl3"
COH = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
CASF_DIR = f"{REPO}/base/_casf"


def _rho_over_seeds(seed_rows, keep=None):
    """seed_rows: list of {pid:(pred,y)}; keep: optional pid set → per-seed ρ list."""
    out = []
    for d in seed_rows:
        items = [(p, v) for p, v in d.items() if (keep is None or p in keep)]
        if len(items) < 3:
            continue
        y = np.array([v[1] for _, v in items]); yh = np.array([v[0] for _, v in items])
        out.append(spearmanr(yh, y).correlation)
    return out


def cl3_seed_preds(cl3_tag):
    seeds = []
    for f in sorted(glob.glob(f"{CL3_DIR}/scatter_*_lp_edrscc_v2_cl123_{cl3_tag}_cl3dump_seed*.csv")):
        d = {r["pid"].strip().lower(): (float(r["y_pred"]), float(r["y_true"]))
             for r in csv.DictReader(open(f))}
        seeds.append(d)
    return seeds


def casf_seed_preds(casf_run, ep):
    seeds = []
    for f in sorted(glob.glob(f"{CASF_DIR}/{casf_run}_e{ep}_corr5_casf2016_preds_seed*.csv")):
        d = {r["pid"].strip().lower(): (float(r["pred"]), float(r["y"]))
             for r in csv.DictReader(open(f))}
        seeds.append(d)
    return seeds


def full_agg(casf_run, ep):
    """FULL = lp_edrscc_v2 (1320) aggregate: per-seed test_spearman from the watcher's probe CSV."""
    pat = (f"{VOX}/dataset/data/pdbbind/results/probe_results_e{ep}_v5_lp_edrscc_v2split_"
           f"loss-mse-corr-w5_{casf_run}_e{ep}_lp_edrscc_v2_msecorr5.csv")
    fs = glob.glob(pat)
    if not fs:
        return []
    return [float(r["test_spearman"]) for r in csv.DictReader(open(fs[0])) if r.get("test_spearman")]


def load_pids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def fmt(vals):
    if not vals:
        return "   —   "
    return f"{st.mean(vals):.3f}±{(st.pstdev(vals) if len(vals) > 1 else 0):.3f}"


def panel(cl3_tag, casf_run, ep, label=None):
    cl3 = cl3_seed_preds(cl3_tag)
    n60 = load_pids(f"{COH}/cl123_test_novel60.txt")
    n30 = load_pids(f"{COH}/cl123_test_novel30.txt")
    casf = casf_seed_preds(casf_run, ep)
    clean = set(r["pid"].strip().lower() for r in csv.DictReader(open(f"{VOX}/splits/casf2016_clean.csv")))
    row = {
        "FULL":      full_agg(casf_run, ep),
        "CL3":       _rho_over_seeds(cl3),
        "CL3-ID60":  _rho_over_seeds(cl3, n60),
        "CL3-ID30":  _rho_over_seeds(cl3, n30),
        "CASF":      _rho_over_seeds(casf),
        "CASF-clean": _rho_over_seeds(casf, clean),
    }
    lab = label or f"{casf_run} e{ep}"
    ns = {"FULL": 1320, "CL3": 733, "CL3-ID60": 454, "CL3-ID30": 262, "CASF": 214, "CASF-clean": 92}
    return lab, row, ns


if __name__ == "__main__":
    cl3_tag, casf_run, ep = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else None
    lab, row, ns = panel(cl3_tag, casf_run, ep, label)
    cols = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF", "CASF-clean"]
    print(f"{'':30} " + " ".join(f"{c}(N{ns[c]})".rjust(16) for c in cols))
    print(f"{lab:30} " + " ".join(fmt(row[c]).rjust(16) for c in cols))
