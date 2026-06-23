#!/usr/bin/env python
"""build_clean_results.py — fill results.html from the probe-result CSVs.

Idempotent + re-runnable: reads probe_results_*.csv from the pdbbind results dir,
computes mean±std (3 seeds) per (encoder, split, metric), and rewrites the
data-key cells / status pills / deltas / best-highlights in results.html
in place. Cells whose CSV is not present yet stay as TBD ("—").

Also copies the consumed CSVs into this folder (260625/) per request.

Run:  python notebook/html/260625/build_clean_results.py
"""
import glob
import os
import re
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data/pdbbind/results"
HTML = os.path.join(HERE, "results.html")
EPOCH, VV = 99, "v5"

# encoder tag (from --tag at probe time) per logical row
TAGS = {
    "coords": "otf_coords_mask050",
    "cdg":    "otf_cdg_mask050",          # plain-ViT C+D+G == Table-2 "vit" row
    "cvit":   "otf_cdg_channelvit_mask050",
    "cha":    "otf_cdg_chamae_mask050",
}


def csv_path(tag, split):
    """lp_edrscc → ..._lp_edrscc_v2split_<tag>.csv (canonical = Kd/Ki-only v2);
    lp(raw) → ..._<tag>.csv (no split token)."""
    if split == "lp_edrscc":
        return f"{RES}/probe_results_e{EPOCH}_{VV}_lp_edrscc_v2split_{tag}.csv"
    return f"{RES}/probe_results_e{EPOCH}_{VV}_{tag}.csv"


def load(tag, split):
    p = csv_path(tag, split)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def ms(df, col, dp=3):
    return f"{df[col].mean():.{dp}f}±{df[col].std(ddof=0):.{dp}f}"


# ── gather all results into a nested dict: data[row][split] = df ────────────────
data = {row: {sp: load(tag, sp) for sp in ("lp_edrscc", "lp")} for row, tag in TAGS.items()}

# ── HTML cell edits ────────────────────────────────────────────────────────────
html = open(HTML).read()


def set_cell(html, key, inner, best=False):
    """Replace the inner text of <td ... data-key="key"> and swap tbd→num/best class."""
    pat = re.compile(r'(<td class=")([^"]*)("(?:\s+[^>]*)?\s+data-key="' + re.escape(key) + r'"[^>]*>)(.*?)(</td>)', re.S)
    def repl(m):
        toks = [t for t in m.group(2).split() if t not in ("tbd", "num", "best")]
        toks.append("best" if best else "num")
        return f'{m.group(1)}{" ".join(toks)}{m.group(3)}{inner}{m.group(5)}'
    new, n = pat.subn(repl, html)
    if n == 0:
        print(f"  [warn] cell key not found: {key}")
    return new


def set_pill(html, key, text, cls):
    pat = re.compile(r'(<span class="pill )[a-z]+(" data-key="' + re.escape(key) + r'">)(.*?)(</span>)')
    new, n = pat.subn(lambda m: f'{m.group(1)}{cls}{m.group(2)}{text}{m.group(4)}', html)
    if n == 0:
        print(f"  [warn] pill key not found: {key}")
    return new


def fill_rows(html, rows, prefix, splits=("edrscc", "lp")):
    """rows = list of (row_key, html_keybase). Fill rho/r/rmse + best per (split, metric)."""
    # gather means for best-detection
    for sp_html, sp_csv in zip(splits, ("lp_edrscc", "lp")):
        for metric, col, better in (("rho", "test_spearman", "max"),
                                    ("r", "test_pearson", "max"),
                                    ("rmse", "test_rmse", "min")):
            means = {}
            for row_key, kb in rows:
                df = data[row_key][sp_csv]
                if df is not None:
                    means[row_key] = df[col].mean()
            best_row = None
            if means:
                best_row = (max if better == "max" else min)(means, key=means.get)
            for row_key, kb in rows:
                df = data[row_key][sp_csv]
                key = f"{prefix}_{kb}_{sp_html}_{metric}"
                if df is None:
                    continue
                inner = ms(df, col)
                if metric == "rho" and sp_csv == "lp":
                    inner += f' <span class="nt">n={int(df["n_test"].iloc[0])}</span>'
                html = set_cell(html, key, inner, best=(row_key == best_row and len(means) > 1))
    return html


# Table 1 — modality (coords vs cdg)
html = fill_rows(html, [("coords", "coords"), ("cdg", "cdg")], "p1")
# Table 2 — architecture (vit==cdg, cvit, cha)
html = fill_rows(html, [("cdg", "vit"), ("cvit", "cvit"), ("cha", "cha")], "p2")

# deltas (cdg − coords) on test ρ
for sp_html, sp_csv in (("edrscc", "lp_edrscc"), ("lp", "lp")):
    c, d = data["coords"][sp_csv], data["cdg"][sp_csv]
    if c is not None and d is not None:
        delta = d["test_spearman"].mean() - c["test_spearman"].mean()
        html = set_cell(html, f"p1_delta_{sp_html}_rho", f"{delta:+.3f}")

# n in lp_edrscc header
if data["coords"]["lp_edrscc"] is not None:
    n = int(data["coords"]["lp_edrscc"]["n_test"].iloc[0])
    html = re.sub(r'(data-key="n_edrscc">)[^<]*(</span>)', rf'\g<1>n={n}\2', html)

# status pills: DONE (probe CSV present) > PRETRAINED (e99 ckpt, probe pending)
# > RUNNING (exp dir exists) > QUEUED (nothing yet).
EXPS = "/home/shpark/prj-denovo/VoxBind/voxbind/exps"
pill_map = {
    "st_p1_coords": ("coords", "260622_plinder_otf_coords_mask050_pretrain"),
    "st_p1_cdg":    ("cdg",    "260622_plinder_otf_cdg_mask050_pretrain"),
    "st_p2_cvit":   ("cvit",   "260622_plinder_otf_cdg_channelvit_mask050_pretrain"),
    "st_p2_cha":    ("cha",    "260622_plinder_otf_cdg_chamae_mask050_pretrain"),
}
for pk, (row, exp) in pill_map.items():
    d = os.path.join(EXPS, exp)
    if data[row]["lp_edrscc"] is not None or data[row]["lp"] is not None:
        html = set_pill(html, pk, "DONE", "done")
    elif os.path.exists(os.path.join(d, "checkpoint_e0099.pth.tar")):
        html = set_pill(html, pk, "PRETRAINED · PROBING", "run")
    elif os.path.isdir(d):
        html = set_pill(html, pk, "RUNNING", "run")
    else:
        html = set_pill(html, pk, "QUEUED", "queue")

# ── Phase-3 autoresearch trials (Table 3) ──────────────────────────────────────
P3_TRIALS = {
    "t1": "260622_ar_vit_t1_atombias",   "t2": "260622_ar_vit_t2_mask060",
    "t3": "260622_ar_vit_t3_dw03",       "t4": "260622_ar_vit_t4_dw10",
    "t5": "260622_ar_vit_t5_warmup10",   "t6": "260622_ar_vit_t6_rope",
    "t7": "260622_ar_vit_t7_mask075",
}
base_df = data["cdg"]["lp_edrscc"]
base_rho = base_df["test_spearman"].mean() if base_df is not None else None
p3_df = {tk: load(exp, "lp_edrscc") for tk, exp in P3_TRIALS.items()}
p3_rho = {tk: df["test_spearman"].mean() for tk, df in p3_df.items() if df is not None}
all_rho = dict(p3_rho)
if base_rho is not None:
    all_rho["base"] = base_rho
best_k = max(all_rho, key=all_rho.get) if all_rho else None
if base_rho is not None:
    html = set_cell(html, "p3_base_rho", f"{base_rho:.3f}", best=(best_k == "base"))
for tk, exp in P3_TRIALS.items():
    df = p3_df[tk]
    d = os.path.join(EXPS, exp)
    if df is not None:
        html = set_cell(html, f"p3_{tk}_rho", ms(df, "test_spearman"), best=(best_k == tk))
        if base_rho is not None:
            html = set_cell(html, f"p3_{tk}_delta", f"{p3_rho[tk] - base_rho:+.3f}")
        html = set_pill(html, f"p3_{tk}_st", "DONE", "done")
    elif os.path.exists(os.path.join(d, "checkpoint_e0099.pth.tar")):
        html = set_pill(html, f"p3_{tk}_st", "PROBING", "run")
    elif os.path.isdir(d):
        html = set_pill(html, f"p3_{tk}_st", "RUNNING", "run")
    else:
        html = set_pill(html, f"p3_{tk}_st", "QUEUED", "queue")

# add .nt style if missing
if ".nt{" not in html:
    html = html.replace("</style>", "  .nt{font-size:10.5px;color:var(--ink-soft);font-weight:400}\n</style>")

open(HTML, "w").write(html)

# ── single consolidated results CSV (replaces per-run copies) ──────────────────
import csv as _csv
CONSOLIDATED = os.path.join(HERE, "clean_results.csv")
P3_META = {
    "t1": ("T1 atom_biased", "mask_strategy=atom_biased"), "t2": ("T2 mask0.6", "mask_ratio=0.6"),
    "t3": ("T3 dens-wt0.3", "density+gradmag wt 0.1->0.3"), "t4": ("T4 dens-wt1.0", "density+gradmag wt 0.1->1.0"),
    "t5": ("T5 dens-warmup", "density wt cosine warmup 10ep"), "t6": ("T6 rope3d", "pos_encoding=rope3d"),
    "t7": ("T7 mask0.75", "mask_ratio=0.75"),
}

def _agg(df, **meta):
    if df is None:
        return None
    r = dict(meta)
    r["n_train"], r["n_test"], r["n_seeds"] = int(df["n_train"].iloc[0]), int(df["n_test"].iloc[0]), len(df)
    for col, key in (("test_spearman", "test_rho"), ("test_pearson", "test_r"),
                     ("test_rmse", "test_rmse"), ("test_mae", "test_mae")):
        r[f"{key}_mean"] = round(float(df[col].mean()), 4)
        r[f"{key}_std"] = round(float(df[col].std(ddof=0)), 4)
    return r

summary = []
for phase, label, arch, mod, rowkey in (
    ("1", "coords-only", "ViT", "C", "coords"),
    ("1+2", "C+D+G", "ViT", "C+D+G", "cdg"),
    ("2", "C+D+G ChannelViT", "ChannelViT", "C+D+G", "cvit"),
    ("2", "C+D+G ChA-MAEViT", "ChA-MAEViT", "C+D+G", "cha"),
):
    for sp in ("lp_edrscc", "lp"):
        r = _agg(data[rowkey][sp], phase=phase, label=label, architecture=arch, modality=mod, knob="-", split=sp)
        if r:
            summary.append(r)
for tk in P3_TRIALS:
    lbl, knob = P3_META[tk]
    r = _agg(p3_df[tk], phase="3", label=lbl, architecture="ViT", modality="C+D+G", knob=knob, split="lp_edrscc")
    if r:
        summary.append(r)

cols = ["phase", "label", "architecture", "modality", "knob", "split", "n_train", "n_test", "n_seeds",
        "test_rho_mean", "test_rho_std", "test_r_mean", "test_r_std", "test_rmse_mean", "test_rmse_std",
        "test_mae_mean", "test_mae_std"]
with open(CONSOLIDATED, "w", newline="") as fcsv:
    w = _csv.DictWriter(fcsv, fieldnames=cols)
    w.writeheader()
    w.writerows(summary)

# remove any previously-copied per-run CSVs (now consolidated into clean_results.csv)
for old in glob.glob(os.path.join(HERE, "probe_results_*.csv")):
    os.remove(old)

print(f"filled results.html; wrote {len(summary)} rows -> {CONSOLIDATED}")
for row in TAGS:
    for sp in ("lp_edrscc", "lp"):
        df = data[row][sp]
        if df is not None:
            print(f"  {row:7s} {sp:10s} n_test={int(df['n_test'].iloc[0]):5d} "
                  f"ρ={ms(df,'test_spearman')} r={ms(df,'test_pearson')} RMSE={ms(df,'test_rmse')}")
