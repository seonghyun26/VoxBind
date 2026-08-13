#!/usr/bin/env python3
"""build_results.py — assemble notebook/html/results.html (results only, no analysis).

Section 1 — Binding-affinity regression: method-comparison table with a GROUPED-COLUMN
  layout over four split variants (lp_edrscc_v2, +CL1, +CL1+CL2, +CL1+CL2+CL3 — the
  progressively-cleaner LP-PDBBind no-leak tiers), each reporting Pearson r / Spearman ρ /
  RMSE (mean ± std, 3 seeds). Cells fill in as the per-split campaign completes; not-yet-run
  cells show as "running". Reuses METHODS/colors from 260715/build_appendixB_bar.py for the
  v2 column + the v2 headline bar chart, and reads CheapNet per-split JSONs live.
  Adds HonestAffinity (arXiv 2606.03422 — leak-aware ESM-2 sequence baseline) as a new row.
Section 2 — De novo drug design: Table 2 (+ heavy-atom count column) + the Vina PNG figure,
  extracted verbatim from 260715_meeting.html.

    python notebook/html/build_results.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # notebook/html
REPO = os.path.dirname(os.path.dirname(HERE))                     # VoxBind

# Heavy-atom counts (Avg, Med) per generated molecule, one entry per Table-2 row in order.
# Rows 0-7 (prior methods) are from the VoxBind paper (Table 1, arXiv:2405.03961).
# Rows 8-10 (our reproduced VoxBind σ0.9/σ1.0 + Ours) are None -> rendered "—": the full
# 79-pocket generation output isn't retained in this checkout. To fill: replace the None
# with (avg, med) once the numbers are ready, then re-run this script.
ATOM_COUNTS = [
    (22.8, 21.5),   # Reference
    (17.6, 16.0),   # AR
    (17.7, 15.0),   # Pocket2Mol
    (24.0, 23.0),   # DiffSBDD
    (24.2, 24.0),   # TargetDiff
    (20.9, 21.0),   # DecompDiff
    (23.4, 24.0),   # VoxBind σ=0.9  (paper)
    (21.7, 22.0),   # VoxBind σ=1.0  (paper)
    None,           # VoxBind σ=0.9  (reproduced, 79 density pockets) — TODO fill
    None,           # VoxBind σ=1.0  (reproduced) — TODO fill
    None,           # Ours (frozen C+D+G σ=0.9) — TODO fill
]


def add_atom_column(table):
    """Append a '# Atoms (Avg/Med)' column group to the extracted de novo Table 2."""
    if "# Atoms" in table:                                        # idempotent
        return table
    thead_end = table.index("</thead>")
    head, body = table[:thead_end], table[thead_end:]
    head = re.sub(r'(<tr class="grp">.*?)</tr>',
                  r'\1<th class="div-major" colspan="2"># Atoms<br>(heavy)</th></tr>',
                  head, count=1, flags=re.S)
    head = re.sub(r'(<tr class="sub">.*?)</tr>',
                  r'\1<th class="div-major">Avg</th><th>Med</th></tr>', head, count=1, flags=re.S)
    cnt = [0]

    def proc(m):
        row = m.group(0)
        if 'colspan="17"' in row:                                # section-band rows
            return row.replace('colspan="17"', 'colspan="19"')
        if "col-method" not in row:
            return row
        i = cnt[0]; cnt[0] += 1
        val = ATOM_COUNTS[i] if i < len(ATOM_COUNTS) else None
        if val is not None:
            a, md = val
            cells = (f'<td class="metric div-major"><span class="val">{a:.1f}</span></td>'
                     f'<td class="metric"><span class="val">{md:.1f}</span></td>')
        else:
            cells = ('<td class="metric div-major"><span class="sd">&mdash;</span></td>'
                     '<td class="metric"><span class="sd">&mdash;</span></td>')
        return re.sub(r'</tr>\s*$', cells + '</tr>', row, flags=re.S)

    body = re.sub(r'<tr\b.*?</tr>', proc, body, flags=re.S)
    return head + body


# Our DecompDiff reproductions (N=100 pockets × 10 mols, Vina full). The two rows live in the
# "Our reproduction" block with reproduced VoxBind/Ours; the paper DecompDiff row stays above:
#   • ref-informed = ref_prior / golden-prior (closest analog to the paper's reference priors).
#     BENCHMARK-FAITHFUL 260729: 25 samples/pocket, exh 32, 98/100 pockets (2 dropped — pockets
#     81/92 had pathologically large molecules that don't dock in reasonable time even at exh 8).
#     Vina Score/Min/Dock −5.14/−6.00/−7.05, High aff 49.2%, QED 0.50, SA 0.66, Diversity 0.85,
#     #atoms 20.9. vs paper Ours −6.04/−7.09/−8.43, High aff 71%, QED 0.43, SA 0.60. Our reference
#     ligands dock at −7.44 (paper −7.26) → pipeline calibrated; the ~1.4 Vina Dock gap is real
#     generation quality, not a docking artifact (public ckpt makes reference-level binders, not
#     beat-reference). Supersedes the earlier 10-sample −6.98.
#   • ref-free = subpocket prior + num_atoms_mode=prior (the fair pocket-only comparison to VoxBind):
#     Vina Dock −5.83/−6.02, #atoms 16.1/15 (undersized → weaker docking).
# High aff. = % of gen mols out-docking their pocket's reference; Diversity = 1-mean pairwise Tanimoto.
# Sim.ref not emitted by the eval → "—". Medians for High aff/Diversity are single aggregates → "—".
_DECOMP_REPRO_ROW = """
          <tr>
            <td class="col-method">DecompDiff <span class="tag repro">reproduced</span><span style="display:block;font-size:11px;color:#7a8699">ref-informed &middot; golden-prior &middot; 25/pocket &middot; exh32 &middot; N=98</span></td>
            <td class="metric div-major"><span class="val">&minus;5.14</span></td>
            <td class="metric"><span class="val">&minus;5.20</span></td>
            <td class="metric div-minor"><span class="val">&minus;6.00</span></td>
            <td class="metric"><span class="val">&minus;5.98</span></td>
            <td class="metric div-minor"><span class="val">&minus;7.05</span></td>
            <td class="metric"><span class="val">&minus;7.17</span></td>
            <td class="metric div-major"><span class="val">49.2</span></td>
            <td class="metric"><span class="val">50.0</span></td>
            <td class="metric div-minor"><span class="val">0.50</span></td>
            <td class="metric"><span class="val">0.49</span></td>
            <td class="metric div-minor"><span class="val">0.66</span></td>
            <td class="metric"><span class="val">0.65</span></td>
            <td class="metric div-minor"><span class="val">0.85</span></td>
            <td class="metric"><span class="val">0.87</span></td>
            <td class="metric div-major"><span class="sd">&mdash;</span></td>
            <td class="metric"><span class="sd">&mdash;</span></td>
            <td class="metric div-major"><span class="val">20.9</span></td>
            <td class="metric"><span class="val">21.0</span></td>
          </tr>
          <tr>
            <td class="col-method">DecompDiff <span class="tag repro">reproduced</span><span style="display:block;font-size:11px;color:#7a8699">ref-free &middot; subpocket-prior &middot; our run &middot; N=100</span></td>
            <td class="metric div-major"><span class="val">&minus;2.48</span></td>
            <td class="metric"><span class="val">&minus;3.51</span></td>
            <td class="metric div-minor"><span class="val">&minus;4.44</span></td>
            <td class="metric"><span class="val">&minus;4.70</span></td>
            <td class="metric div-minor"><span class="val">&minus;5.83</span></td>
            <td class="metric"><span class="val">&minus;6.02</span></td>
            <td class="metric div-major"><span class="sd">&mdash;</span></td>
            <td class="metric"><span class="sd">&mdash;</span></td>
            <td class="metric div-minor"><span class="val">0.52</span></td>
            <td class="metric"><span class="val">0.51</span></td>
            <td class="metric div-minor"><span class="val">0.68</span></td>
            <td class="metric"><span class="val">0.68</span></td>
            <td class="metric div-minor"><span class="sd">&mdash;</span></td>
            <td class="metric"><span class="sd">&mdash;</span></td>
            <td class="metric div-major"><span class="sd">&mdash;</span></td>
            <td class="metric"><span class="sd">&mdash;</span></td>
            <td class="metric div-major"><span class="val">16.1</span></td>
            <td class="metric"><span class="val">15.0</span></td>
          </tr>"""


def inject_decompdiff_repro(table):
    """Insert reproduced DecompDiff at the start of the Our reproduction block (idempotent)."""
    if "tag repro" in table:
        return table
    old_band = "Our reproduction · 79 x-ray-density pockets"
    new_band = ("Our reproduction · DecompDiff 98/100 pockets · "
                "VoxBind/Ours 79 x-ray-density pockets")
    table = table.replace(old_band, new_band)
    i = table.find(new_band)
    if i < 0:
        return table
    j = table.index("</tr>", i) + len("</tr>")
    return table[:j] + _DECOMP_REPRO_ROW + table[j:]


def mark_voxbind_reproduced(table):
    """Use the reproduced tag for our two rerun VoxBind rows; paper rows remain prior."""
    return table.replace(
        'VoxBind <span class="tag supervised">baseline</span>',
        'VoxBind <span class="tag repro">reproduced</span>',
    )


def strip_repeated_table_units(table):
    """Keep units in Table 4 headers only (e.g. High aff. (%)), not every value cell."""
    return re.sub(r'(<span class="val">[^<]*?)%(</span>)', r'\1\2', table)


sys.path.insert(0, os.path.join(HERE, "260715"))
import build_appendixB_bar as bar                                 # noqa: E402  (METHODS, svg(), legend())
import build_regression_baseline as reg                           # noqa: E402  (read_cheapnet())
import build_denovo_vina_chart as denovo_chart                    # noqa: E402  (Matplotlib Figure 4)

DOC715 = os.path.join(HERE, "260715", "260715_meeting.html")
OUT = os.path.join(HERE, "results.html")

# ── the four split variants (scheme, display label, test-N) ───────────────────
SPLITS = [
    ("lp_edrscc_v2",       "lp_edrscc_v2",   1320),
    ("lp_edrscc_v2_cl1",   "+CL1",           1166),
    ("lp_edrscc_v2_cl12",  "+CL1+CL2",       1149),
    ("lp_edrscc_v2_cl123", "+CL1+CL2+CL3",    733),
]

# train / val / test counts per scheme (from splits/MANIFEST.json) — shown in the Table 1 caption.
COUNTS = {
    "lp_edrscc_v2":       (3850, 817, 1320),
    "lp_edrscc_v2_cl1":   (2721, 680, 1166),
    "lp_edrscc_v2_cl12":  (2643, 659, 1149),
    "lp_edrscc_v2_cl123": (1559, 410,  733),
}

# ── method metadata (v2 values + colors from bar.METHODS; + HonestAffinity) ───
V2 = {name: {"r": r, "rho": rho, "rmse": rmse}
      for (name, color, flag, r, rho, rmse) in bar.METHODS}
COLOR = {name: color for (name, color, *_ ) in bar.METHODS}
FLAG = {name: flag for (name, color, flag, *_ ) in bar.METHODS}
COLOR["HonestAffinity"] = bar.PALETTE[11]     # distinct cyan from the shared palette
FLAG["HonestAffinity"] = "new"
COLOR["IPNet (frozen)"] = bar.PALETTE[13]      # lavender — IPDiff BAPNet, frozen (leaked)
COLOR["IPNet (scratch)"] = bar.PALETTE[14]     # pink — IPDiff BAPNet, from scratch (clean)
COLOR["Nesso-1"] = "#00838f"                   # teal — Nesso-1 cofolding (leaked: PDBbind-trained)
# seq/SMILES DTA baselines (amber family, matching the seq tier) + CPES (3D curvature/PES)
COLOR["DeepDTA"] = "#c98a3a"; COLOR["MolTrans"] = "#a86a26"
COLOR["CPES"] = "#6f5aa8"
for _n in ("DeepDTA", "MolTrans", "CPES"):
    FLAG[_n] = "new"

TYPE = {  # display tag per method — three categories, distinctly colored
    "HBGSA": "supervised", "EGNN": "supervised", "EGNN + TargetDiff": "supervised",
    "GET": "supervised", "ProFSA": "pretrained", "CheapNet": "supervised", "BindNet": "supervised",
    "HonestAffinity": "supervised", "AEV-PLIG": "supervised", "DSMBind": "zero-shot",
    "IPNet (frozen)": "pretrained", "IPNet (scratch)": "supervised", "Nesso-1": "zero-shot",
    "C": "pretrained", "C+D+G": "pretrained", "C+D+G +corr": "pretrained",
    "DeepDTA": "supervised", "MolTrans": "supervised",
    "CPES": "supervised",
}
TAGCLASS = {"supervised": "supervised", "pretrained": "pretrained", "zero-shot": "zeroshot"}

# Input modality — what the method actually consumes. "seq" = protein sequence + ligand SMILES
# only (no 3D coordinates); "3d" = needs the 3D structure (voxel/graph/pocket/energy). Methods not
# listed here default to 3D, including our voxel C/C+D+G, CPES, the graph baselines, ProFSA, DSMBind.
MODALITY = {
    "Nesso-1": "seq", "HonestAffinity": "seq",
    "DeepDTA": "seq", "MolTrans": "seq",
}  # default → "3d"
DENSITY_OURS = {"C+D+G", "C+D+G +corr"}   # our voxel+density models → their own top information tier

# Methods with NO official author checkpoint — the whole model was re-trained from scratch on our
# data (faithful reimplementation or vendored code). Flagged "re-trained" for transparency. NOT
# flagged: methods that use official pretrained weights (ProFSA / IPNet-frozen = frozen official
# encoder + probe; DSMBind / Nesso-1 / Boltz-2 = official checkpoint, zero-shot or as-is).
RETRAINED = {"GET", "EGNN", "EGNN + TargetDiff", "CheapNet", "HBGSA", "BindNet",
             "AEV-PLIG", "HonestAffinity", "IPNet (scratch)", "DeepDTA", "MolTrans"}

# Methods whose prediction is NOT on the pK scale → RMSE is not meaningful (report ρ only):
# DSMBind = zero-shot binding energy (arbitrary units/sign); IPNet (frozen) = frozen features + head
# on a SUM readout whose magnitude isn't pK-calibrated. Their RMSE cell renders "n/a".
CORR_ONLY = {"DSMBind", "IPNet (frozen)"}

# Methods whose correlation SIGN is arbitrary → report the magnitude |r|, |ρ| (marked with †). A
# zero-shot binding-energy score anti-correlates with pK by construction, so on the zero-shot holdout
# DSMBind's r/ρ come out negative; the trained-probe v2/CL evals are already positive (abs is a no-op).
ABS_CORR = {"DSMBind"}


def absmetric(name, metric, v):
    """|value| for the r/ρ of sign-arbitrary (zero-shot energy) methods; rmse & others pass through."""
    if v is None or v[0] is None or metric == "rmse" or name not in ABS_CORR:
        return v
    return (abs(v[0]), v[1])


# Input-information tiers shown as a merged (rowspan) first column, ordered seq → 3D → +density,
# with all LEAKED methods pulled into their own group pinned to the very BOTTOM of every table
# (regardless of their real modality) so they read as an excluded reference block, not a ranked tier.
CAT_ORDER = {"seq": 0, "3d": 1, "ours": 2, "leaked": 3}
CAT_LABEL = {"seq": "seq/SMILES", "3d": "3D structure",
             "ours": "3D structure<br>+&#8202;density <b>(ours)</b>",
             "leaked": "leaked<br>(excluded)"}
TYPE_ORDER = {"supervised": 0, "zero-shot": 1, "pretrained": 2}
OURS_ORDER = {"C+D+G": 0, "C+D+G +corr": 1}


def modality_category(name):
    """Input tier without applying benchmark-specific leakage labels."""
    if MODALITY.get(name, "3d") == "seq":
        return "seq"
    if name in DENSITY_OURS:
        return "ours"
    return "3d"


def category3(name):
    if name in LEAKED:                    # leaked → own group at the bottom, before modality matters
        return "leaked"
    return modality_category(name)


def cat_sort_key(name, within):
    """Table order: input tier, supervised→zero-shot→pretrained, then the table's own order.

    C is our coordinates-only encoder, so keep it at the bottom of the 3D-coordinate tier.
    """
    cat = category3(name)
    coords_only_last = cat == "3d" and name == "C"
    stable_within = OURS_ORDER.get(name, within) if cat == "ours" else within
    return (CAT_ORDER[cat], coords_only_last,
            TYPE_ORDER.get(TYPE.get(name, "supervised"), 9), stable_within)


def cat_merged_rows(items, category_fn=category3):
    """items: [(method, row_content_html)] already sorted so same-category rows are consecutive.
    Emits <tr>s with a plain-text, rowspan-merged category cell as the first column."""
    out, i, n = [], 0, len(items)
    while i < n:
        cat = category_fn(items[i][0]); j = i
        while j < n and category_fn(items[j][0]) == cat:
            j += 1
        for k in range(i, j):
            m, content = items[k]
            head = (f'<td class="col-modality cat-{cat}" rowspan="{j - i}">{CAT_LABEL[cat]}</td>'
                    if k == i else "")
            out.append(f"<tr>{head}{content}</tr>")
        i = j
    return "\n          ".join(out)

# Methods trained on affinity data that OVERLAPS our PDBbind test set → their scores are a
# leaked ceiling, not clean generalization. Flagged with a "leaked" badge AND excluded from the
# best/second ranking (they'd otherwise win on memorization). Nesso-1: trained on PDBbind+BindingDB+
# ChEMBL (all of lp_edrscc_v2 is in it). IPNet (frozen): BAPNet supervised on PDBbind-v2016 affinity.
LEAKED = {"Nesso-1", "IPNet (frozen)"}
for _name in LEAKED:
    FLAG[_name] = "leaked"

# Frozen pretrained model each method leans on (orthogonal to the supervised/pretrained/zero-shot
# TYPE tag — e.g. HonestAffinity is supervised-trained but on top of a frozen ESM backbone).
# ESM users get a highlighted badge; other frozen pretrained backbones a neutral one. Methods
# trained from scratch on our data (GET/CheapNet/AEV-PLIG/HBGSA/BindNet/EGNN/IPNet-scratch) → none.
BACKBONE = {
    "HonestAffinity": ("ESM-2 650M",   "esm"),       # frozen per-residue ESM-2-650M seq embeddings
    "DSMBind":        ("ESM-2 3B",     "esm"),        # frozen per-residue ESM-2-3B pocket embeddings
    "ProFSA":         ("ProFSA-pre",   "backbone"),   # frozen pretrained pocket encoder
    "IPNet (frozen)": ("affinity-pre", "backbone"),   # BAPNet supervised-pretrained on PDBbind affinity
    "C":              ("voxel-MAE",    "backbone"),    # our SSL-pretrained voxel ViT (frozen)
    "C+D+G":          ("voxel-MAE",    "backbone"),
    "C+D+G +corr":    ("voxel-MAE",    "backbone"),    # same encoder; probe head = MSE + Pearson-aux (λ5)
    "Nesso-1":        ("ESM-2",        "esm"),         # cofolding trunk on ESM-2 protein embeddings
}

# Methods whose CL cells are literature/external numbers we do NOT reproduce per-split
# → render "—" instead of "running". Empty: the whole grid is being reproduced this campaign.
EXTERNAL = set()

# Methods whose v2 value in bar.METHODS is a PAPER number (not run on our ED+RSCC split) —
# suppress it (show "running") until we train them on our splits (base/get, model_type EGNN).
PAPER_ONLY = {"EGNN", "EGNN + TargetDiff"}

# Planned baseline with no official checkpoint/result on our cohorts. Keep every missing cell
# explicitly TBA (rather than the generic live-campaign "running" state used by Table 1a).
PLANNED_TBA = {"CPES"}

# method render order = bar.METHODS order (HonestAffinity, IPNet frozen/scratch, and Nesso-1 are
# first-class entries in bar.METHODS so Table 1 and the bar charts share one source).
ORDER = [name for (name, *_ ) in bar.METHODS]

# ── CASF-2016 evaluation columns (all methods trained on lp_edrscc_v2 TRAIN, tested on CASF) ──
# leaky = full CASF-2016 core (214 ED-avail, 90 in v2-train → inflated); nontrain = the 124 held out.
# Results in base/_casf/{key}.json with {"leaky":{pearson:{mean,std},...}, "nontrain":{...}}.
# clean = CASF2016-clean: the 92 core-set complexes NOT in our lp_edrscc_v2 train OR val — the only
# truly held-out CASF subset (leaky 214 has 90 train + 32 val overlap; nontrain 124 still has 32 val).
# Consolidated to ONE honest subset: the clean-92 core — CASF-2016 complexes held out from
# BOTH the downstream affinity split (lp_edrscc_v2 train/val) AND the champion's SSL pretraining
# (PLINDER-v2; only 5 CASF overlap it, all already excluded by the lp_edrscc_v2 filter). So clean-92
# is truly unseen by our method end-to-end. (leaky-214 / nontrain-124 dropped.)
CASF_COLS = [("clean", "CASF-2016 (held-out ED, clean-92)", 92)]
CASF_KEY = {"C": "C_100m_mask075_coords", "C+D+G": "CDG_100m_mask075",
            "C+D+G +corr": "CDG_100m_mask075_corr5", "DSMBind": "DSMBind", "GET": "GET", "EGNN": "EGNN",
            "EGNN + TargetDiff": "EGNN_TD", "CheapNet": "CheapNet", "BindNet": "BindNet",
            "AEV-PLIG": "AEV", "HBGSA": "HBGSA", "ProFSA": "ProFSA", "HonestAffinity": "HonestAffinity",
            "IPNet (frozen)": "IPNet_frozen", "IPNet (scratch)": "IPNet_retrain", "Nesso-1": "Nesso",
            "DeepDTA": "DeepDTA", "MolTrans": "MolTrans"}


def load_casf(method, which):
    """CASF result for a method → {'r','rho','rmse'} or None. which ∈ {'leaky','nontrain'}."""
    key = CASF_KEY.get(method)
    if not key:
        return None
    path = os.path.join(REPO, "base", "_casf", f"{key}.json")
    if not os.path.exists(path):
        return None
    import json as _j
    blk = _j.load(open(path)).get(which)
    if not isinstance(blk, dict):
        return None
    g = lambda k: (blk[k]["mean"], blk[k].get("std")) if isinstance(blk.get(k), dict) else None
    r, rho, rmse = g("pearson"), g("spearman"), g("rmse")
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


_RESULTS_DIR = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "results")

# VoxBind frozen-encoder probe CSVs per (method, split) — C = coords-only 100M ChannelViT MATCHED to
# C+D+G (260723_ar_cvit_100m_v2_mask075_coords, e49): same 100M/PLINDER-v2/mask075 recipe, density
# channels dropped → C vs C+D+G is now an apples-to-apples density ablation (Δρ +0.048 on v2),
# C+D+G = best recipe: 100M ChannelViT, PLINDER-v2 pretrain, 75% mask (260705_ar_cvit_100m_v2_mask075,
# epoch 49) — ρ 0.644 on v2, re-probed consistently on all CL tiers + CASF (supersedes g742 0.637).
PROBE_CSV = {
    "C": {
        "lp_edrscc_v2":       "probe_results_e49_v5_lp_edrscc_v2split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "lp_edrscc_v2_cl1":   "probe_results_e49_v5_lp_edrscc_v2_cl1split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "lp_edrscc_v2_cl12":  "probe_results_e49_v5_lp_edrscc_v2_cl12split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "lp_edrscc_v2_cl123": "probe_results_e49_v5_lp_edrscc_v2_cl123split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "lba60":              "probe_results_e49_v5_lba60split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "lba30":              "probe_results_e49_v5_lba30split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "atom3d_lba60_edrscc_v2": "probe_results_e49_v5_atom3d_lba60split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "atom3d_lba30_edrscc_v2": "probe_results_e49_v5_atom3d_lba30split_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "atom3d_lba60_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba60_v22cleansplit_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "atom3d_lba30_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba30_v22cleansplit_260723_ar_cvit_100m_v2_mask075_coords.csv",
        "clean_ed_v1_indep":  "probe_results_e49_v5_clean_ed_v1_indepsplit_260723_ar_cvit_100m_v2_mask075_coords.csv",
    },
    "C+D+G": {
        "lp_edrscc_v2":       "probe_results_e49_v5_lp_edrscc_v2split_260705_ar_cvit_100m_v2_mask075.csv",
        "lp_edrscc_v2_cl1":   "probe_results_e49_v5_lp_edrscc_v2_cl1split_260705_ar_cvit_100m_v2_mask075.csv",
        "lp_edrscc_v2_cl12":  "probe_results_e49_v5_lp_edrscc_v2_cl12split_260705_ar_cvit_100m_v2_mask075.csv",
        "lp_edrscc_v2_cl123": "probe_results_e49_v5_lp_edrscc_v2_cl123split_260705_ar_cvit_100m_v2_mask075.csv",
        "lba60":              "probe_results_e49_v5_lba60split.csv",
        "lba30":              "probe_results_e49_v5_lba30split.csv",
        "atom3d_lba60_edrscc_v2": "probe_results_e49_v5_atom3d_lba60split.csv",
        "atom3d_lba30_edrscc_v2": "probe_results_e49_v5_atom3d_lba30split.csv",
        "atom3d_lba60_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba60_v22cleansplit.csv",
        "atom3d_lba30_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba30_v22cleansplit.csv",
        "clean_ed_v1_indep":  "probe_results_e49_v5_clean_ed_v1_indepsplit_260705_ar_cvit_100m_v2_mask075.csv",
    },
    # C+D+G with the probe HEAD trained as MSE + Pearson-correlation aux (λ=5) — same 100M encoder,
    # head-only change (260806). Slightly higher r/ρ and lower RMSE than the MSE-head C+D+G.
    "C+D+G +corr": {
        "lp_edrscc_v2":       "probe_results_e49_v5_lp_edrscc_v2split_loss-mse-corr-w5.csv",
        "lp_edrscc_v2_cl1":   "probe_results_e49_v5_lp_edrscc_v2_cl1split_loss-mse-corr-w5.csv",
        "lp_edrscc_v2_cl12":  "probe_results_e49_v5_lp_edrscc_v2_cl12split_loss-mse-corr-w5.csv",
        "lp_edrscc_v2_cl123": "probe_results_e49_v5_lp_edrscc_v2_cl123split_loss-mse-corr-w5.csv",
        "lba60":              "probe_results_e49_v5_lba60split_loss-mse-corr-w5.csv",
        "lba30":              "probe_results_e49_v5_lba30split_loss-mse-corr-w5.csv",
        "atom3d_lba60_edrscc_v2": "probe_results_e49_v5_atom3d_lba60split_loss-mse-corr-w5.csv",
        "atom3d_lba30_edrscc_v2": "probe_results_e49_v5_atom3d_lba30split_loss-mse-corr-w5.csv",
        "atom3d_lba60_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba60_v22cleansplit_loss-mse-corr-w5.csv",
        "atom3d_lba30_edrscc_v2_v22clean": "probe_results_e49_v5_atom3d_lba30_v22cleansplit_loss-mse-corr-w5.csv",
        "clean_ed_v1_indep":  "probe_results_e49_v5_clean_ed_v1_indepsplit_loss-mse-corr-w5.csv",
    },
}


# Per-(method, split) result JSONs written by the CL campaign runs, formatted with the split.
# Short-tag paths use v2/cl1/cl12/cl123; full-tag paths use the scheme name. Checked before the
# v2/paper fallback so cells auto-fill as each baseline's run lands; a missing file → None.
SHORT = {"lp_edrscc_v2": "v2", "lp_edrscc_v2_cl1": "cl1",
         "lp_edrscc_v2_cl12": "cl12", "lp_edrscc_v2_cl123": "cl123",
         "lba60": "lba60", "lba30": "lba30",
         "atom3d_lba60_edrscc_v2": "atom3d_lba60_edrscc_v2",
         "atom3d_lba30_edrscc_v2": "atom3d_lba30_edrscc_v2",
         "atom3d_lba60_edrscc_v2_v22clean": "atom3d_lba60_edrscc_v2_v22clean",
         "atom3d_lba30_edrscc_v2_v22clean": "atom3d_lba30_edrscc_v2_v22clean",
         "clean_ed_v1_indep": "clean_ed_v1_indep"}
METHOD_PATH = {
    "GET":               lambda s: f"base/get/_edrscc/results_GET_{SHORT[s]}.json",
    "EGNN":              lambda s: f"base/get/_edrscc/results_EGNN_{SHORT[s]}.json",
    "EGNN + TargetDiff": lambda s: f"base/get/_edrscc/results_EGNN_TD_{SHORT[s]}.json",
    "BindNet":           lambda s: f"base/bindnet/_edrscc/results_{s}.json",
    "ProFSA":            lambda s: f"base/profsa/results/profsa_{s}.json",
    "AEV-PLIG":          lambda s: f"base/aevplig/results/aevplig_{s}.json",
    "HBGSA":             lambda s: f"base/hbgsa/results/hbgsa_{SHORT[s]}.json",
    "HonestAffinity":    lambda s: f"base/honestaffinity/results/results_{s}.json",
    "DSMBind":           lambda s: f"base/dsmbind/results/dsmbind_probe_{s}.json",
    "IPNet (frozen)":    lambda s: f"base/ipdiff/_edrscc/results_IPNet_frozen_{s}.json",
    "IPNet (scratch)":   lambda s: f"base/ipdiff/_edrscc/results_IPNet_retrain_{s}.json",
    "Nesso-1":           lambda s: f"base/nesso/_edrscc/results_Nesso_{s}.json",
    "DeepDTA":           lambda s: f"base/dta/result/DeepDTA_{s}.json",
    "MolTrans":          lambda s: f"base/dta/result/MolTrans_{s}.json",
}


def _read_result(relpath):
    """Format-tolerant result reader → {'r':(m,sd),'rho':(m,sd),'rmse':(m,sd)} or None.

    Handles the four schemas the campaign emits:
      A  {"pearson":{"mean","std"}, ...}                       (GET/EGNN, HBGSA, BindNet, Honest)
      B  {"per_seed_mean_std":{"test_pearson":[m,s], ...}}     (AEV-PLIG)
      D  {"mean_std":{"test_pearson":[m,s], ...}}              (ProFSA)
      C  {"mean_pearson","std_pearson", ...}                  (DSMBind probe)
    """
    path = os.path.join(REPO, relpath)
    if not os.path.exists(path):
        return None
    import json as _j
    d = _j.load(open(path))
    if (isinstance(d.get("pearson"), dict) and "mean" in d["pearson"]) or \
       (isinstance(d.get("spearman"), dict) and "mean" in d["spearman"]):        # A (rho-only ok)
        g = lambda k: (d[k]["mean"], d[k].get("std")) if isinstance(d.get(k), dict) and "mean" in d[k] else None
        r, rho, rmse = g("pearson"), g("spearman"), g("rmse")
    elif isinstance(d.get("per_seed_mean_std"), dict) or isinstance(d.get("mean_std"), dict):   # B / D
        m = d.get("per_seed_mean_std") or d.get("mean_std")
        g = lambda k: (m[k][0], m[k][1]) if isinstance(m.get(k), (list, tuple)) and len(m[k]) >= 2 else None
        r, rho, rmse = g("test_pearson"), g("test_spearman"), g("test_rmse")
    elif "mean_pearson" in d:                                                    # C
        g = lambda mk, sk: (d[mk], d.get(sk)) if mk in d else None
        r, rho, rmse = g("mean_pearson", "std_pearson"), g("mean_spearman", "std_spearman"), g("mean_rmse", "std_rmse")
    else:
        return None
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


def _probe_csv(fname):
    """Read a probe_results CSV (per-seed rows) → {'r','rho','rmse'} mean±std over seeds."""
    import csv as _csv
    import statistics as _st
    path = os.path.join(_RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    rows = list(_csv.DictReader(open(path)))

    def col(c):
        vs = [float(r[c]) for r in rows if r.get(c) not in (None, "", "nan")]
        if not vs:
            return None
        return (sum(vs) / len(vs), _st.pstdev(vs) if len(vs) > 1 else 0.0)

    r, rho, rmse = col("test_pearson"), col("test_spearman"), col("test_rmse")
    if r is None and rho is None:
        return None
    return {"r": r, "rho": rho, "rmse": rmse}


def load(method, split):
    """Per-method-per-split result loader → {'r':(m,sd),'rho':(m,sd),'rmse':(m,sd)} or None.

    Live campaign JSONs/CSVs fill as each run lands; else fall back to the v2 column from
    bar.METHODS (our runs), or None (→ "running" placeholder). Paper v2 (EGNN rows) suppressed.
    """
    if split in ("lp_test_novel60", "lp_test_novel30"):       # LP protein-novelty test subsets:
        # SAME lp_edrscc_v2 model/features, TEST masked by max train seq-id (<60% / <30%). Every
        # method with cached per-complex v2 preds is re-scored by base/_casf/score_novel_subsets.py
        # (no retraining) → schema-A JSON. Missing file → None (TBA).
        safe_m = "".join(c if c.isalnum() else "_" for c in method)
        return _read_result(f"base/_casf/novel_subsets/{safe_m}__{split}.json")
    if method == "CheapNet":
        return reg.read_cheapnet(split)                       # results_{split}.json (all 4 done)
    if method in PROBE_CSV:                                    # VoxBind C / C+D+G frozen-encoder probe
        fn = PROBE_CSV[method].get(split)
        return _probe_csv(fn) if fn else None
    if method in METHOD_PATH:                                  # campaign JSONs fill as they land
        d = _read_result(METHOD_PATH[method](split))
        if d:
            return d
    if method in PAPER_ONLY:
        return None                                           # suppress paper v2 until run on our split
    if split == "lp_edrscc_v2":
        return V2.get(method)                                 # v2 column from bar.METHODS (our runs)
    return None                                               # CL cells: pending this campaign


# ── % within N log units (|pred − exp pK| < N) — a TerraBind/Nesso-style success-rate metric,
# computed from per-complex predictions on the v2 TEST set. Only methods with saved pK-scale
# predictions are covered; others store only summary metrics (GET/CheapNet/HBGSA/BindNet/
# HonestAffinity/IPNet/EGNN) or an uncalibrated energy (DSMBind) → shown as "—" (need a pred dump).
# Probe rows (C, C+D+G) are single-seed (scatter seed 0); the others are their reported predictor.
PRED_FILES = {  # method: (relpath, true_col, pred_col)
    "C":        ("voxbind/dataset/data/pdbbind/results/scatter/scatter_c_100m_mask075_coords_v2.csv", "y_true", "y_pred"),
    "C+D+G":    ("voxbind/dataset/data/pdbbind/results/scatter/scatter_cdg_100m_mask075_v2.csv", "y_true", "y_pred"),
    "Nesso-1":  ("base/nesso/_edrscc/predictions_v2.csv", "pK", "pred_pIC50"),
    "AEV-PLIG": ("base/aevplig/results/aevplig_edrscc_preds.csv", "truth", "pred_ensemble"),
}


def pct_within(method, thresh=2.0):
    """Percent of v2 test points with |pred − exp pK| < thresh (and N), or None if no preds."""
    import csv as _csv
    spec = PRED_FILES.get(method)
    if not spec:
        return None
    path = os.path.join(REPO, spec[0])
    if not os.path.exists(path):
        return None
    tc, pc = spec[1], spec[2]
    n = w = 0
    for r in _csv.DictReader(open(path)):
        try:
            t, p = float(r[tc]), float(r[pc])
        except (KeyError, ValueError, TypeError):
            continue
        n += 1
        w += abs(p - t) < thresh
    return (100.0 * w / n, n) if n else None


def within_table():
    """Compact TerraBind-style table: % of v2 test within 2 (and 1) log units, methods with preds."""
    rows = []
    for m in ORDER:
        w2 = pct_within(m, 2.0)
        if not w2:
            continue
        w1 = pct_within(m, 1.0)
        rows.append((m, w2[0], w1[0] if w1 else None, w2[1]))
    rows.sort(key=lambda x: -x[1])
    best2 = max((r[1] for r in rows), default=None)
    body = []
    for m, p2, p1, n in rows:
        b = ' style="font-weight:700"' if p2 == best2 else ""
        p1s = f"{p1:.1f}" if p1 is not None else "&mdash;"
        body.append(f'<tr>{method_cell(m)}'
                    f'<td class="metric"{b}>{p2:.1f}</td><td class="metric">{p1s}</td>'
                    f'<td class="metric"><span class="sd">{n}</span></td></tr>')
    return "\n          ".join(body)


def best_per_col():
    """best method per column among present values (r/ρ max, RMSE min); LP splits + CASF."""
    best = {}
    for split, _, _ in SPLITS:
        for metric, hi in (("r", True), ("rho", True), ("rmse", False)):
            vals = [(m, load(m, split)[metric][0]) for m in ORDER
                    if m not in LEAKED and load(m, split) and load(m, split).get(metric) and load(m, split)[metric][0] is not None]
            if vals:
                best[(split, metric)] = (max if hi else min)(vals, key=lambda x: x[1])[0]
    for which, _, _ in CASF_COLS:
        for metric, hi in (("r", True), ("rho", True), ("rmse", False)):
            vals = [(m, load_casf(m, which)[metric][0]) for m in ORDER
                    if m not in LEAKED and load_casf(m, which) and load_casf(m, which).get(metric) and load_casf(m, which)[metric][0] is not None]
            if vals:
                best[("casf", which, metric)] = (max if hi else min)(vals, key=lambda x: x[1])[0]
    return best


def rank_per_col():
    """{(split, metric): (best_method, second_method)} over the LP split columns.
    r/ρ higher = better, RMSE lower = better. second is None if <2 values present."""
    out = {}
    for split, _, _ in SPLITS:
        for metric, hi in (("r", True), ("rho", True), ("rmse", False)):
            vals = [(m, load(m, split)[metric][0]) for m in all_methods()
                    if m not in LEAKED and load(m, split) and load(m, split).get(metric) and load(m, split)[metric][0] is not None]
            vals.sort(key=lambda x: x[1], reverse=hi)
            if vals:
                out[(split, metric)] = (vals[0][0], vals[1][0] if len(vals) > 1 else None)
    return out


def slice_between(text, start, end, frm=0, include_end=True):
    i = text.index(start, frm)
    j = text.index(end, i) + (len(end) if include_end else 0)
    return text[i:j], j


def extract_from_715():
    doc = open(DOC715, encoding="utf-8").read()
    css = doc[doc.index("<style>"):doc.index("</style>") + len("</style>")]
    # De novo Table 2 — the <table class="results"> after the "Table 2" title
    k = doc.index("Table 2 &nbsp;")
    table2, _ = slice_between(doc, '<table class="results">', "</table>", frm=k)
    sub2, _ = slice_between(doc, '<p class="table-sub">', "</p>", frm=k)
    # De novo Figure 1 — the base64 Vina PNG (find the <img ...> after the "Figure 1" title)
    f = doc.index("Figure 1 &nbsp;")
    img, _ = slice_between(doc, "<img ", ">", frm=f)
    return css, table2, sub2, img


# ── Section 1: grouped affinity table over the four split variants ────────────
def cell(v, best=False, div=False):
    cls = "metric" + (" div-major" if div else "") + (" best" if best else "")
    if v is None or v[0] is None:
        return f'<td class="{cls}"><span class="na">&mdash;</span></td>'
    m, sd = v
    sds = f' <span class="sd">±{sd:.3f}</span>' if sd is not None else ""
    return f'<td class="{cls}"><span class="val">{m:.3f}</span>{sds}</td>'


def pending_cell(external, div=False):
    cls = "metric" + (" div-major" if div else "")
    inner = '<span class="na">&mdash;</span>' if external else '<span class="tbd">running</span>'
    return f'<td class="{cls}">{inner}</td>'


def metric_cell(v, rank, divcls):
    """One metric td. rank: True|'best' → best (bold), 'second' → underline, else none."""
    cls = "metric" + (f" {divcls}" if divcls else "")
    if rank is True or rank == "best":
        cls += " best"
    elif rank == "second":
        cls += " second"
    if v is None or v[0] is None:
        return f'<td class="{cls}"><span class="na">&mdash;</span></td>'
    m, sd = v
    sds = f' <span class="sd">±{sd:.3f}</span>' if sd is not None else ""
    return f'<td class="{cls}"><span class="val">{m:.3f}</span>{sds}</td>'


def pending2(divcls):
    cls = "metric" + (f" {divcls}" if divcls else "")
    return f'<td class="{cls}"><span class="tbd">running</span></td>'


def tba_cell(divcls):
    """A not-yet-run cell — explicit TBA so the reader can tell the method is pending, not omitted."""
    cls = "metric" + (f" {divcls}" if divcls else "")
    return f'<td class="{cls}"><span class="tbd">TBA</span></td>'


def all_methods():
    """Every Section-1.1 baseline (bar.METHODS order) plus newer baseline additions, so the
    held-out tables list ALL methods — with TBA for any not yet run on the new consolidated subset."""
    extra = [m for m in ("DeepDTA", "MolTrans", "CPES") if m not in ORDER]
    return list(ORDER) + extra


def method_cell(name, dagger=False, show_leaked=True, pretrain_overlap=False):
    typ = TYPE.get(name, "supervised")
    tagc = TAGCLASS[typ]
    color = COLOR.get(name, "#888")
    flag = FLAG.get(name, False)
    border = ";border:2px solid #000" if flag is True else ""   # best marker only; new rows not highlighted
    sw = (f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
          f'background:{color};margin-right:7px;vertical-align:middle{border}"></span>')
    dg = ('<sup class="dg" title="zero-shot binding-energy score; correlation sign is arbitrary '
          '&mdash; |r| / |&rho;| reported">&dagger;</sup>' if dagger else "")
    bb = BACKBONE.get(name)
    bb_html = (f'<span class="tag {bb[1]}" title="frozen pretrained backbone">{bb[0]}</span>'
               if bb else "")
    leak_html = ('<span class="tag leaked" title="trained on affinity data overlapping our PDBbind '
                 'test set &mdash; leaked ceiling, excluded from best/second ranking">leaked</span>'
                 if show_leaked and name in LEAKED else "")
    overlap_html = ('<span class="tag leaked" title="exact CASF structures occur in the released '
                    'self-supervised pretraining corpus">pretrain overlap</span>'
                    if pretrain_overlap else "")
    retr_html = ('<span class="tag retrained" title="no official author checkpoint &mdash; the whole '
                 'model was re-trained from scratch on our data">re-trained</span>'
                 if name in RETRAINED else "")
    return (f'<td class="col-method">{sw}{name}{dg}<span class="tag {tagc}">{typ}</span>'
            f'{bb_html}{retr_html}{leak_html}{overlap_html}</td>')


# CASP16 Stage-2 shortlist: representative sequence/3D baselines plus our coordinate-only and
# density-enabled probes. Values intentionally stay TBA until one common evaluable cohort is fixed.
CASP16_METHODS = ["Nesso-1", "C", "C+D+G"]
CASP16_KEY = {"Nesso-1": "CASP16_Nesso", "C": "CASP16_C", "C+D+G": "CASP16_CDG"}


def casp16_table_head():
    return (
        '<tr class="grp"><th class="col-modality" rowspan="2">Input</th>'
        '<th class="col-method" rowspan="2">Method</th>'
        '<th class="div-major" colspan="4">CASP16 chymase (L1000) &mdash; held-out affinity'
        '<span class="nsub">density-common subset &middot; IC50, single congeneric series</span></th></tr>\n'
        '          <tr class="sub"><th class="div-major">Pearson&nbsp;<i>r</i></th>'
        '<th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th><th>N</th></tr>'
    )


def load_casp16(name):
    """Return {'r':[mean,std],'rho':...,'rmse':...,'n':N} from base/_casp16/*.json, or None."""
    key = CASP16_KEY.get(name)
    if not key:
        return None
    path = os.path.join(REPO, "base", "_casp16", f"{key}.json")
    if not os.path.exists(path):
        return None
    import json as _j
    d = _j.load(open(path)).get("casp16", {})
    def pair(k):
        m = d.get(k)
        return [m["mean"], m["std"]] if m else None
    return {"r": pair("pearson"), "rho": pair("spearman"), "rmse": pair("rmse"), "n": d.get("n")}


def casp16_rows():
    """CASP16 chymase-17 rows (Nesso-1 / C / C+D+G). No best-highlight: n=17 single series is tied/noisy."""
    items = []
    for name in CASP16_METHODS:            # seq (Nesso-1) then 3D (C, C+D+G); cat_merged_rows groups Input
        d = load_casp16(name)
        tds = [method_cell(name, show_leaked=True)]
        for k, metric in enumerate(("r", "rho", "rmse")):
            divcls = "div-major" if k == 0 else ""
            if d is None or d.get(metric) is None:
                tds.append(tba_cell(divcls))
            else:
                tds.append(metric_cell(d[metric], False, divcls))
        n = d.get("n") if d else None
        n_html = str(n) if n else '<span class="tbd">TBA</span>'
        tds.append(f'<td class="metric">{n_html}</td>')
        items.append((name, "".join(tds)))
    return cat_merged_rows(items, modality_category)


# ── LBA-style protein-sequence-novelty splits (Table 1b) ─────────────────────
LBA_METHODS = all_methods()   # identical baseline universe in Tables 1a/1b; missing runs render TBA
LBA_COLS = [   # base LP, then LP TEST masked by protein novelty (same model/features, no retrain), then CleanSplit
    ("lp_edrscc_v2", "LP-PDBBind", "1320"),
    ("lp_test_novel60", "LP &middot; novel protein<br>test id &lt; 60% to train", "813"),
    ("lp_test_novel30", "LP &middot; novel protein<br>test id &lt; 30% to train", "453"),
    ("clean_ed_v1_indep",  "CleanSplit leak-free &middot; CASF-2016", "109"),
]


def load_lba_table():
    import json as _j
    p = os.path.join(REPO, "base", "_casf", "lba_table.json")
    return _j.load(open(p)) if os.path.exists(p) else {}


def lba_table_head():
    grp = ['<tr class="grp"><th class="col-modality" rowspan="2">Input</th>'
           '<th class="col-method" rowspan="2">Method</th>']
    for _, label, n in LBA_COLS:
        grp.append(f'<th class="div-major" colspan="3">{label}<span class="nsub">N&nbsp;=&nbsp;{n}</span></th>')
    grp.append("</tr>")
    sub = ['<tr class="sub">']
    for _ in LBA_COLS:
        sub.append('<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>')
    sub.append("</tr>")
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def lba_rows():
    """All Table-1 baselines as rows (same as §1.1). First column = the lp_edrscc_v2 value via load()
    (identical to Table 1a); LBA & CleanSplit columns fill from each method's result file as runs land, else TBA."""
    best = {}
    for key, _, _ in LBA_COLS:
        present = [(m, load(m, key)) for m in LBA_METHODS]
        present = [(m, d) for m, d in present
                   if m not in LEAKED and d and d.get("rho") and d["rho"][0] is not None]
        if present:
            best[key] = max(present, key=lambda x: x[1]["rho"][0])[0]
    order = sorted(LBA_METHODS, key=lambda n: cat_sort_key(n, LBA_METHODS.index(n)))  # match Table 1 modality order
    items = []
    for m in order:
        tds = [method_cell(m, show_leaked=True)]
        for key, _, _ in LBA_COLS:
            d = load(m, key)
            for k, metric in enumerate(("r", "rho", "rmse")):
                divcls = "div-major" if k == 0 else ""
                if d is None or d.get(metric) is None or d[metric][0] is None:
                    tds.append(tba_cell(divcls))
                else:
                    rank = "best" if (metric == "rho" and best.get(key) == m) else None
                    tds.append(metric_cell(d[metric], rank, divcls))
        items.append((m, "".join(tds)))
    return cat_merged_rows(items)


def lba_bar_svg():
    """Table-1b values as dataset rows x metric columns, using the Table-1 method order."""
    order = sorted(LBA_METHODS, key=lambda n: cat_sort_key(n, LBA_METHODS.index(n)))
    panels = [
        ("r", "Test Pearson r", 0.35, 0.85, [0.40, 0.50, 0.60, 0.70, 0.80], True),
        ("rho", "Test Spearman &rho;", 0.35, 0.85, [0.40, 0.50, 0.60, 0.70, 0.80], True),
        ("rmse", "Test RMSE", 1.30, 1.90, [1.30, 1.50, 1.70, 1.90], False),
    ]
    row_labels = {
        "lp_edrscc_v2": ("LP-PDBBind", "base"),
        "lp_test_novel60": ("Novel &lt;60%", "to train"),
        "lp_test_novel30": ("Novel &lt;30%", "to train"),
        "clean_ed_v1_indep": ("CleanSplit", "CASF-2016"),
    }
    W, header_h, row_h, row_gap, bottom = 1130, 40, 160, 14, 16
    H, ml, mr, col_gap = header_h + len(LBA_COLS) * row_h + (len(LBA_COLS) - 1) * row_gap + bottom, 160, 16, 24
    plot_w = (W - ml - mr - col_gap * (len(panels) - 1)) / len(panels)
    out = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;margin:0 auto" '
        'font-family="-apple-system,Segoe UI,Roboto,sans-serif" role="img" '
        'aria-label="Table 1b generalization metrics across LP-PDBBind, protein-novel subsets, and CleanSplit">',
        '<title>Generalization under leakage control</title>',
        '<desc>Columns are Pearson correlation, Spearman correlation, and RMSE; rows are the four '
        'Table 1b cohorts. Bars show available method means and whiskers show standard deviation.</desc>',
    ]

    for pi, (_, title, _, _, _, _) in enumerate(panels):
        panel_x = ml + pi * (plot_w + col_gap)
        out.append(f'<text x="{panel_x + plot_w / 2:.1f}" y="20" text-anchor="middle" '
                   f'font-size="11.5" font-weight="700" fill="#1c2433">{title}</text>')

    for ri, (cohort, _, n_test) in enumerate(LBA_COLS):
        row_y = header_h + ri * (row_h + row_gap)
        plot_top, plot_base = row_y + 10, row_y + 132
        main_label, sub_label = row_labels[cohort]
        out.append(f'<text x="{ml-30}" y="{row_y+55:.1f}" text-anchor="end" font-size="10.5" '
                   f'font-weight="700" fill="#566072">{main_label}</text>')
        out.append(f'<text x="{ml-30}" y="{row_y+69:.1f}" text-anchor="end" font-size="8.5" '
                   f'fill="#7a8699">{sub_label}</text>')
        out.append(f'<text x="{ml-30}" y="{row_y+83:.1f}" text-anchor="end" font-size="8.5" '
                   f'fill="#9aa3b2">N = {n_test}</text>')

        for pi, (metric, _, vmin, vmax, ticks, higher_better) in enumerate(panels):
            panel_x = ml + pi * (plot_w + col_gap)

            def y(value):
                return plot_base - (value - vmin) / (vmax - vmin) * (plot_base - plot_top)

            for tick in ticks:
                yy = y(tick)
                out.append(f'<line x1="{panel_x:.1f}" y1="{yy:.1f}" x2="{panel_x+plot_w:.1f}" y2="{yy:.1f}" '
                           'stroke="#e6e9ef" stroke-width="1"/>')
                out.append(f'<text x="{panel_x-5:.1f}" y="{yy+3:.1f}" text-anchor="end" font-size="8" '
                           f'fill="#9aa3b2">{tick:.2f}</text>')

            inner = 8
            pitch = (plot_w - inner * 2) / len(order)
            bar_width = min(9.0, pitch * 0.58)
            values = []
            for mi, name in enumerate(order):
                data = load(name, cohort)
                value = data.get(metric) if data else None
                if not value or value[0] is None:
                    continue
                mean = value[0]
                sd = value[1] if value[1] is not None else 0.0
                values.append((mi, name, mean, sd, name in LEAKED))

            ranked = [item for item in values if not item[4]]
            best_i = None
            if ranked:
                best_i = (max if higher_better else min)(ranked, key=lambda item: item[2])[0]

            for mi, name, mean, sd, leaked in values:
                cx = panel_x + inner + pitch * (mi + 0.5)
                bar_y = max(plot_top, min(plot_base, y(mean)))
                height = max(0.0, plot_base - bar_y)
                opacity = ' opacity="0.38"' if leaked else ""
                stroke = ' stroke="#000" stroke-width="1.5"' if mi == best_i else ""
                out.append(f'<rect x="{cx-bar_width/2:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                           f'height="{height:.1f}" rx="1.5" fill="{COLOR[name]}"{stroke}{opacity} '
                           f'data-method="{name}" data-cohort="{cohort}" data-metric="{metric}">'
                           f'<title>{name} · {main_label}: {mean:.3f} ± {sd:.3f}</title></rect>')
                yhi = max(plot_top, min(plot_base, y(mean + sd)))
                ylo = max(plot_top, min(plot_base, y(mean - sd)))
                out.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" '
                           f'stroke="#566072" stroke-width="0.8"{opacity} data-whisker="true" '
                           f'data-ymin="{plot_top:.1f}" data-ymax="{plot_base:.1f}"/>')
                if mi == best_i:
                    label_y = max(plot_top + 9, yhi - 3)
                    out.append(f'<text x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                               f'font-size="8.5" font-weight="700" fill="#1c2433">{mean:.2f}</text>')

            out.append(f'<line x1="{panel_x:.1f}" y1="{plot_base:.1f}" '
                       f'x2="{panel_x+plot_w:.1f}" y2="{plot_base:.1f}" '
                       'stroke="#aeb6c4" stroke-width="1.2"/>')

    out.append('</svg>')
    return "\n      ".join(out)


def lba_bar_legend():
    """Legend for methods with at least one value in Table 1b; TBA-only methods stay in the table."""
    order = sorted(LBA_METHODS, key=lambda n: cat_sort_key(n, LBA_METHODS.index(n)))
    plotted = [name for name in order if any(load(name, cohort) for cohort, _, _ in LBA_COLS)]
    methods = [(name, COLOR[name], "leaked" if name in LEAKED else False,
                (None, None), (None, None), (None, None)) for name in plotted]
    return bar.legend(methods=methods, leaked_names=LEAKED)


def affinity_table_head():
    grp = ['<tr class="grp"><th class="col-modality" rowspan="2">Input</th>'
           '<th class="col-method" rowspan="2">Method</th>']
    for _, label, n in SPLITS:
        grp.append(f'<th class="div-major" colspan="3">{label}<span class="nsub">N&nbsp;=&nbsp;{n}</span></th>')
    grp.append("</tr>")
    sub = ['<tr class="sub">']
    for _ in SPLITS:
        sub.append('<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>')
    sub.append("</tr>")
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def affinity_rows():
    rank = rank_per_col()
    methods = all_methods()
    order = sorted(methods, key=lambda n: cat_sort_key(n, methods.index(n)))
    items = []
    for name in order:
        external = name in EXTERNAL
        tds = [method_cell(name)]
        for split, _, _ in SPLITS:
            d = load(name, split)
            for k, metric in enumerate(("r", "rho", "rmse")):
                divcls = "div-major" if k == 0 else ""
                if d is None or d.get(metric) is None or d[metric][0] is None:
                    if name in PLANNED_TBA:
                        tds.append(tba_cell(divcls))
                    else:
                        tds.append(pending2(divcls) if not external else metric_cell(None, None, divcls))
                else:
                    bm, sm = rank.get((split, metric), (None, None))
                    rk = "best" if name == bm else ("second" if name == sm else None)
                    tds.append(metric_cell(d[metric], rk, divcls))
        items.append((name, "".join(tds)))
    return cat_merged_rows(items)


def casf_table_head():
    grp = ['<tr class="grp"><th class="col-modality" rowspan="2">Input</th>'
           '<th class="col-method" rowspan="2">Method</th>']
    for _, label, n in CASF_COLS:
        grp.append(f'<th class="div-major" colspan="3">{label}<span class="nsub">N&nbsp;=&nbsp;{n}</span></th>')
    grp.append("</tr>")
    sub = ['<tr class="sub">']
    for _ in CASF_COLS:
        sub.append('<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>')
    sub.append("</tr>")
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def casf_rows():
    """CASF appendix table — methods sorted by non-train ρ (honest), leaky + non-train × r/ρ/RMSE."""
    best = best_per_col()
    methods = all_methods()                       # ALL 1.1 baselines; TBA for those not yet on clean-92
    def cleanrho(m):
        d = load_casf(m, "clean") if m in CASF_KEY else None
        return d["rho"][0] if d and d.get("rho") else None
    order = sorted(methods, key=lambda m: cat_sort_key(m, -(cleanrho(m) if cleanrho(m) is not None else -99)))
    items = []
    for name in order:
        tds = [method_cell(name)]
        for which, _, _ in CASF_COLS:
            d = load_casf(name, which) if name in CASF_KEY else None
            for k, metric in enumerate(("r", "rho", "rmse")):
                divcls = "div-major" if k == 0 else ""
                if d is None or d.get(metric) is None or d[metric][0] is None:
                    tds.append(tba_cell(divcls))
                else:
                    # no best-highlight on CASF-2016: clean-92 (n=92) is too small — top methods
                    # are statistically tied, so bolding a "winner" over-reads the noise. 2019 holdout
                    # (Table 3) keeps its ranking. (was: best.get(("casf", which, metric)) == name)
                    tds.append(metric_cell(d[metric], False, divcls))
        items.append((name, "".join(tds)))
    return cat_merged_rows(items)


# ── CASF Table-A1 bar charts — Figure-1 style, values from load_casf() ───────
CASF_PANELS = [
    dict(idx=0, key=0, title="Pearson r",      vmin=0.35, vmax=0.90,
         ticks=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    dict(idx=1, key=1, title="Spearman &rho;", vmin=0.35, vmax=0.90,
         ticks=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    dict(idx=2, key=2, title="RMSE",           vmin=1.00, vmax=2.35,
         ticks=[1.00, 1.25, 1.50, 1.75, 2.00, 2.25]),
]


def casf_bar_svg(which):
    """Figure-1-style r/ρ/RMSE bars for one Table-A1 CASF cohort."""
    present = [m for m in (list(ORDER) + [x for x in CASF_KEY if x not in ORDER])
               if m in CASF_KEY and load_casf(m, "clean")]
    present.sort(key=lambda m: cat_sort_key(m, -(load_casf(m, "clean")["rho"][0] or -1)))
    methods = []
    for name in present:
        d = load_casf(name, which)
        if not d:
            continue

        def val(key):
            item = d.get(key)
            if not item or item[0] is None:
                return (0.0, 0.0)
            return (item[0], item[1] if item[1] is not None else 0.0)

        methods.append((name, COLOR[name], FLAG[name], val("r"), val("rho"), val("rmse")))

    save_m, save_p = bar.METHODS, bar.PANELS
    try:
        bar.METHODS, bar.PANELS = methods, CASF_PANELS
        return bar.svg(rank_labels=True, value_decimals=2)
    finally:
        bar.METHODS, bar.PANELS = save_m, save_p


def casf_bar_charts_block():
    """Two aligned rows for the leaky and non-train cohorts in Table A1."""
    cohorts = [
        ("leaky", "CASF-2016 core", "N = 214 · includes 90 train-overlap complexes"),
        ("nontrain", "CASF-2016 core − train", "N = 124 · exact train members removed"),
    ]
    parts = []
    for which, label, note in cohorts:
        parts.append(
            f'<p class="figure-cap" style="text-align:left;font-weight:700;color:#1c2433;'
            f'margin:16px 0 2px;font-size:13px;">{label} '
            f'<span style="font-weight:400;color:#9aa3b2;">&mdash; {note}</span></p>'
        )
        parts.append(casf_bar_svg(which))
    return "\n      ".join(parts)


def casf_bar_legend():
    """Method-color legend matching the Table-A1 row order."""
    present = [m for m in (list(ORDER) + [x for x in CASF_KEY if x not in ORDER])
               if m in CASF_KEY and load_casf(m, "clean")]
    present.sort(key=lambda m: cat_sort_key(m, -(load_casf(m, "clean")["rho"][0] or -1)))
    methods = [(name, COLOR[name], FLAG[name]) for name in present]
    return bar.legend(methods, LEAKED)


def load_holdout2019_common():
    """Load the identical-92-complex temporal-holdout results used by Table A2b."""
    import json as _j
    path = os.path.join(REPO, "base", "_casf", "_holdout2019_common.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return _j.load(handle)


def holdout2019_common_rows():
    """Rows for the consolidated common-ED-set table, sorted within tiers by ρ. Full-coverage
    methods share one identical common set; partial methods (HBGSA/Nesso, coverage < set) are on
    a subset — flagged with * on N and excluded from the best/second ranking."""
    R = load_holdout2019_common()
    methods = all_methods()                       # list ALL 1.1 baselines; TBA for those not yet run
    rho_mean = lambda m: R[m]["rho"]["mean"] if m in R else None
    order = sorted(methods, key=lambda m: cat_sort_key(m, -(rho_mean(m) if rho_mean(m) is not None else -99)))
    elig = [m for m in R if not R[m].get("partial") and m not in LEAKED]
    best = max((R[m]["rho"]["mean"] for m in elig), default=None)
    items = []
    for m in order:
        if m in R:
            b = R[m]
            def _cell(k):
                if k == "rmse" and m in CORR_ONLY:
                    return ('<td class="metric"><span class="tbd" title="prediction not on the pK '
                            'scale (zero-shot energy / SUM-pool features) &mdash; RMSE not meaningful; '
                            '&rho; is the ranking metric">n/a</span></td>')
                return metric_cell(absmetric(m, k, (b[k]["mean"], b[k]["std"])),
                                   (k == "rho" and b["rho"]["mean"] == best and m in elig),
                                   "div-major" if k == "r" else "")
            cells = "".join(_cell(k) for k in ("r", "rho", "rmse"))
            nc = f'<td class="metric"><span class="sd">{b["n"]}{"*" if b.get("partial") else ""}</span></td>'
        else:
            cells = "".join(tba_cell("div-major" if k == "r" else "") for k in ("r", "rho", "rmse"))
            nc = '<td class="metric"><span class="tbd">TBA</span></td>'
        items.append((m, method_cell(m, dagger=(m in ABS_CORR)) + cells + nc))
    return cat_merged_rows(items)


# ── Merged held-out table: CASF-2016 clean-92 (3 cols) | PDBbind 2019 holdout common (4 cols
# incl. per-method N). Both are held-out, PDBbind-derived evals, so they share one table with two
# column groups (like Table 1's split groups).
def heldout_table_head():
    grp = ['<tr class="grp"><th class="col-modality" rowspan="2">Input</th>'
           '<th class="col-method" rowspan="2">Method</th>'
           '<th class="div-major" colspan="3">CASF-2016<span class="nsub">clean-92 &middot; N&nbsp;=&nbsp;92</span></th>'
           f'<th class="div-major" colspan="4">PDBbind 2019 holdout<span class="nsub">common ED &middot; '
           f'N&nbsp;=&nbsp;{holdout2019_common_n()}</span></th></tr>']
    sub = ['<tr class="sub">'
           '<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>'
           '<th class="div-major">Pearson&nbsp;<i>r</i></th><th>Spearman&nbsp;&rho;</th><th>RMSE&nbsp;&darr;</th>'
           '<th>N</th></tr>']
    return "\n          ".join(grp) + "\n          " + "\n          ".join(sub)


def heldout_rows():
    """One row per method: CASF-2016 clean-92 (no best-highlight — n=92 too small, top methods
    tied) then PDBbind 2019 holdout common (ρ ranked). DSMBind reports |r|/|ρ| (†); CORR_ONLY RMSE
    renders 'n/a'; partial-coverage N marked *; TBA where a method hasn't been run on that subset."""
    R = load_holdout2019_common()
    methods = all_methods()
    crho_m = lambda m: (load_casf(m, "clean") or {}).get("rho", None)
    crho_v = lambda m: crho_m(m)[0] if (m in CASF_KEY and crho_m(m)) else None
    hrho_v = lambda m: R[m]["rho"]["mean"] if m in R else None
    within = lambda m: -(hrho_v(m) if hrho_v(m) is not None
                         else (crho_v(m) if crho_v(m) is not None else -99))
    order = sorted(methods, key=lambda m: cat_sort_key(m, within(m)))
    elig = [m for m in R if not R[m].get("partial") and m not in LEAKED]   # holdout ρ ranking pool
    hbest = max((R[m]["rho"]["mean"] for m in elig), default=None)
    items = []
    for m in order:
        # CASF-2016 clean-92 group (no best-highlight)
        cd = load_casf(m, "clean") if m in CASF_KEY else None
        casf_cells = ""
        for k, metric in enumerate(("r", "rho", "rmse")):
            divcls = "div-major" if k == 0 else ""
            if cd is None or cd.get(metric) is None or cd[metric][0] is None:
                casf_cells += tba_cell(divcls)
            else:
                casf_cells += metric_cell(cd[metric], False, divcls)
        # PDBbind 2019 holdout common group (ρ ranked; DSMBind |·|; CORR_ONLY RMSE n/a; per-method N)
        if m in R:
            b = R[m]

            def _hc(k):
                divcls = "div-major" if k == "r" else ""
                if k == "rmse" and m in CORR_ONLY:
                    return ('<td class="metric"><span class="tbd" title="prediction not on the pK '
                            'scale (zero-shot energy / SUM-pool features) &mdash; RMSE not meaningful; '
                            '&rho; is the ranking metric">n/a</span></td>')
                return metric_cell(absmetric(m, k, (b[k]["mean"], b[k]["std"])),
                                   (k == "rho" and b["rho"]["mean"] == hbest and m in elig), divcls)
            hold_cells = "".join(_hc(k) for k in ("r", "rho", "rmse"))
            hold_cells += (f'<td class="metric"><span class="sd">{b["n"]}'
                           f'{"*" if b.get("partial") else ""}</span></td>')
        else:
            hold_cells = "".join(tba_cell("div-major" if k == "r" else "") for k in ("r", "rho", "rmse"))
            hold_cells += '<td class="metric"><span class="tbd">TBA</span></td>'
        items.append((m, method_cell(m, dagger=(m in ABS_CORR), pretrain_overlap=(m == "ProFSA"))
                      + casf_cells + hold_cells))
    return cat_merged_rows(items)


def load_holdout2019_scalable():
    """Load the SCALABLE-common table (methods that reach the harder misato complexes)."""
    import json as _j
    path = os.path.join(REPO, "base", "_casf", "_holdout2019_scalable.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return _j.load(handle)


def holdout2019_scalable_rows():
    """Rows for the SCALABLE-common head-to-head (n≈346), sorted by ρ."""
    R = load_holdout2019_scalable()
    if not R:
        return ""
    order = sorted(R, key=lambda m: cat_sort_key(m, -R[m]["rho"]))
    best = max((R[m]["rho"] for m in R), default=None)
    items = []
    for m in order:
        b = R[m]
        cells = "".join(metric_cell((b[k], None), (k == "rho" and b["rho"] == best),
                                    "div-major" if k == "r" else "") for k in ("r", "rho", "rmse"))
        items.append((m, method_cell(m) + cells))
    return cat_merged_rows(items)


def holdout2019_scalable_n():
    R = load_holdout2019_scalable()
    return next(iter(R.values()))["n"] if R else 0


def holdout2019_common_n():
    R = load_holdout2019_common()
    ns = [b["n"] for b in R.values() if not b.get("partial")]   # full-coverage common size
    return max(ns) if ns else 0


# ── Table-A2b bar chart — Figure-1 rendering, identical 92 complexes ───────
# Axes (260811) carry every Table-3 row. DSMBind's sign-arbitrary correlation is reported as a
# magnitude (|r|/|ρ|, marked †), so all r/ρ are positive → r/ρ span the IPNet-frozen floor (≈ 0.25)
# up to the leaked Nesso-1 ceiling (≈ 0.69). RMSE spans the pK-scale spread (Nesso 1.52 → MolTrans
# 1.99); the two non-pK methods (DSMBind, IPNet-frozen) contribute no RMSE bar (None).
HOLDOUT2019_COMMON_PANELS = [
    dict(idx=0, key=0, title="Pearson |r|",      vmin=0.20, vmax=0.72,
         ticks=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
    dict(idx=1, key=1, title="Spearman |&rho;|", vmin=0.20, vmax=0.72,
         ticks=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70]),
    dict(idx=2, key=2, title="RMSE",             vmin=1.45, vmax=2.05,
         ticks=[1.50, 1.60, 1.70, 1.80, 1.90, 2.00]),
]


def holdout2019_common_bar_methods():
    """Table-3 values in the same grouped order as the table, with shared method colors.

    Includes ALL Table-3 methods — leaked references (FLAG 'leaked' → faded, de-ranked) and the
    partial Nesso-1 subset — so the figure mirrors the table. RMSE is omitted (None → no bar) for
    CORR_ONLY methods whose prediction is not on the pK scale (matches the table's 'n/a')."""
    results = load_holdout2019_common()
    displayed = set(all_methods())
    order = sorted((method for method in results if method in displayed),
                   key=lambda method: cat_sort_key(method, -results[method]["rho"]["mean"]))
    out = []
    for method in order:
        b = results[method]
        rmse = (None, None) if method in CORR_ONLY else (b["rmse"]["mean"], b["rmse"]["std"])
        out.append((
            method,
            COLOR[method],
            FLAG[method],
            absmetric(method, "r", (b["r"]["mean"], b["r"]["std"])),
            absmetric(method, "rho", (b["rho"]["mean"], b["rho"]["std"])),
            rmse,
        ))
    return out


def holdout2019_common_bar_svg():
    """Figure-1-style r/ρ/RMSE bars for Table A2b's common 92 complexes."""
    methods = holdout2019_common_bar_methods()
    save_m, save_p = bar.METHODS, bar.PANELS
    try:
        bar.METHODS, bar.PANELS = methods, HOLDOUT2019_COMMON_PANELS
        return bar.svg(rank_labels=True, value_decimals=2)
    finally:
        bar.METHODS, bar.PANELS = save_m, save_p


def holdout2019_common_bar_legend():
    """Method-color legend matching the Table-A2b row and bar order."""
    leg = bar.legend(holdout2019_common_bar_methods(), LEAKED)
    for m in ABS_CORR:            # mark |r|/|ρ| methods with † in the legend too
        leg = leg.replace(f"</span> {m}</span>",
                          f'</span> {m}<sup class="dg" title="|r|/|&rho;| reported">&dagger;</sup></span>')
    return leg


def holdout2019_rows():
    """Rows for the 2019 temporal-holdout table (base/_casf/_holdout2019_summary.json), sorted by ρ."""
    import json as _j
    path = os.path.join(REPO, "base", "_casf", "_holdout2019_summary.json")
    if not os.path.exists(path):
        return ""
    R = _j.load(open(path))
    order = sorted(R, key=lambda m: cat_sort_key(m, -(R[m]["rho"][0] if R[m].get("rho") else -1)))
    best = max((R[m]["rho"][0] for m in R if R[m].get("rho")), default=None)
    items = []
    for m in order:
        b = R[m]
        cells = "".join(metric_cell((b[k][0], b[k][1]), (k == "rho" and b["rho"][0] == best),
                                    "div-major" if k == "r" else "") for k in ("r", "rho", "rmse"))
        nc = f'<td class="metric"><span class="sd">{b["n"]}</span></td>'
        items.append((m, method_cell(m) + cells + nc))
    return cat_merged_rows(items)


def casf_leakage_svg():
    """Dumbbell chart: per method, non-train ρ (honest, filled dot) → leaky ρ (inflated, open dot),
    sorted by leakage gap. Red = big leak (≥0.10, memorizers); blue = moderate; grey = negligible."""
    data = []
    for m in ORDER:
        lk, nt = load_casf(m, "leaky"), load_casf(m, "nontrain")
        if lk and nt and lk.get("rho") and nt.get("rho") and lk["rho"][0] is not None and nt["rho"][0] is not None:
            data.append((m, nt["rho"][0], lk["rho"][0]))
    data.sort(key=lambda t: t[2] - t[1], reverse=True)
    X0, X1, TOP, RH = 205, 855, 46, 26
    H = TOP + len(data) * RH + 52
    vmin, vmax = 0.40, 0.86
    def x(r): return X0 + (r - vmin) / (vmax - vmin) * (X1 - X0)
    o = [f'<svg viewBox="0 0 910 {H:.0f}" width="100%" style="max-width:910px;display:block;margin:0 auto" '
         'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    o.append('<text x="455" y="20" font-size="12.5" font-weight="700" fill="#1c2433" text-anchor="middle">'
             'CASF-2016 leakage gap &mdash; non-train (honest) &rarr; leaky (train-overlap), Spearman &rho;</text>')
    ybot = TOP + len(data) * RH
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        xt = x(t)
        o.append(f'<line x1="{xt:.1f}" y1="{TOP-6:.0f}" x2="{xt:.1f}" y2="{ybot:.0f}" stroke="#eceff3" stroke-width="1"/>')
        o.append(f'<text x="{xt:.1f}" y="{ybot+15:.0f}" font-size="9" fill="#9aa3b2" text-anchor="middle">{t:.1f}</text>')
    for i, (m, nt, lk) in enumerate(data):
        y = TOP + i * RH + RH / 2
        gap = lk - nt
        col = "#d64545" if gap >= 0.10 else ("#9aa3b2" if gap < 0.03 else "#3f7fc4")
        o.append(f'<text x="{X0-10:.0f}" y="{y+3:.0f}" font-size="10.5" fill="#1c2433" text-anchor="end">{m}</text>')
        o.append(f'<line x1="{x(nt):.1f}" y1="{y:.1f}" x2="{x(lk):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2.6" opacity=".85"/>')
        o.append(f'<circle cx="{x(nt):.1f}" cy="{y:.1f}" r="4.3" fill="{col}"/>')
        o.append(f'<circle cx="{x(lk):.1f}" cy="{y:.1f}" r="4.3" fill="#fff" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{x(max(lk,nt))+8:.1f}" y="{y+3:.0f}" font-size="9" fill="{col}" font-weight="600">{gap:+.3f}</text>')
    ly = H - 10
    o.append(f'<circle cx="{X0}" cy="{ly-3}" r="4.3" fill="#5b6678"/>'
             f'<text x="{X0+9}" y="{ly}" font-size="9.5" fill="#5b6678">non-train (honest / OOD)</text>')
    o.append(f'<circle cx="{X0+175}" cy="{ly-3}" r="4.3" fill="#fff" stroke="#5b6678" stroke-width="2"/>'
             f'<text x="{X0+184}" y="{ly}" font-size="9.5" fill="#5b6678">leaky (incl. 90 train-overlap)</text>')
    o.append(f'<text x="{X0+400}" y="{ly}" font-size="9.5" fill="#d64545" font-weight="600">red = big leak (&ge;0.10)</text>')
    o.append("</svg>")
    return "\n      ".join(o)


# ── per-tier (CL) bar charts — Figure-1 style, values from load() ─────────────
# Widened common axes shared across the three tiers (so the lowest baselines aren't
# clipped and tiers stay visually comparable); reuses bar.svg()'s exact rendering.
CL_PANELS = [
    dict(idx=0, key=0, title="Test Pearson r",      vmin=0.40, vmax=0.72, ticks=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
    dict(idx=1, key=1, title="Test Spearman &rho;", vmin=0.40, vmax=0.72, ticks=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
    dict(idx=2, key=2, title="Test RMSE",           vmin=1.30, vmax=1.70, ticks=[1.30, 1.40, 1.50, 1.60, 1.70]),
]


def cl_bar_svg(split):
    """Figure-1-style 3-panel (r/ρ/RMSE) bar chart for one CL tier, values from load().

    Same 11 chart methods/colors as Figure 1; renders via bar.svg() with bar.METHODS and
    bar.PANELS temporarily swapped (restored in finally so Figure 1 is unaffected)."""
    methods = []
    for (name, color, flag, *_) in bar.METHODS:
        d = load(name, split)
        if not d:
            continue

        def val(k):
            t = d.get(k)
            if not t or t[0] is None:
                return (0.0, 0.0)
            return (t[0], t[1] if t[1] is not None else 0.0)

        methods.append((name, color, flag, val("r"), val("rho"), val("rmse")))
    save_m, save_p = bar.METHODS, bar.PANELS
    try:
        bar.METHODS, bar.PANELS = methods, CL_PANELS
        return bar.svg(rank_labels=True)
    finally:
        bar.METHODS, bar.PANELS = save_m, save_p


def cl_charts_block():
    """Stacked per-tier bar charts (+CL1, +CL1+CL2, +CL1+CL2+CL3) with a labelled header each."""
    tiers = [("lp_edrscc_v2_cl1", "+CL1"),
             ("lp_edrscc_v2_cl12", "+CL1+CL2"),
             ("lp_edrscc_v2_cl123", "+CL1+CL2+CL3")]
    parts = []
    for scheme, label in tiers:
        tr, va, te = COUNTS[scheme]
        parts.append(
            f'<p class="figure-cap" style="text-align:left;font-weight:700;color:#1c2433;'
            f'margin:16px 0 2px;font-size:13px;">{label} '
            f'<span style="font-weight:400;color:#9aa3b2;">&mdash; test N = {te} '
            f'(train {tr} / val {va})</span></p>')
        parts.append(cl_bar_svg(scheme))
    return "\n      ".join(parts)


def lp_all_tiers_svg():
    """Figure 1: split rows x metric columns, with a divider before CL filtering."""
    methods = list(bar.METHODS)
    methods = ([m for m in methods if m[0] not in LEAKED]
               + [m for m in methods if m[0] in LEAKED])
    panels = [
        ("r", "Test Pearson r", 0.40, 0.72, [0.40, 0.50, 0.60, 0.70], True),
        ("rho", "Test Spearman &rho;", 0.40, 0.72, [0.40, 0.50, 0.60, 0.70], True),
        ("rmse", "Test RMSE", 1.30, 1.70, [1.30, 1.40, 1.50, 1.60, 1.70], False),
    ]
    W, header_h, row_h, row_gap, bottom = 1130, 40, 160, 14, 16
    H, ml, mr, col_gap = header_h + len(SPLITS) * row_h + (len(SPLITS) - 1) * row_gap + bottom, 132, 16, 24
    plot_w = (W - ml - mr - col_gap * (len(panels) - 1)) / len(panels)
    divider_y = header_h + row_h + row_gap / 2
    out = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;margin:0 auto" '
        'font-family="-apple-system,Segoe UI,Roboto,sans-serif" role="img" '
        'aria-label="LP-PDBBind metrics across the unfiltered and three CL-cleaned tiers">',
        '<title>LP-PDBBind metrics across all cleaning tiers</title>',
        '<desc>Columns are metrics and rows are the original split and three CL-cleaned tiers. '
        'A horizontal line separates the original split from the CL tiers.</desc>',
        f'<text x="18" y="{header_h + row_h / 2:.1f}" text-anchor="middle" font-size="10.5" '
        f'font-weight="700" fill="#566072" transform="rotate(-90 18 {header_h + row_h / 2:.1f})">No CL</text>',
        f'<text x="18" y="{divider_y + (H - bottom - divider_y) / 2:.1f}" text-anchor="middle" font-size="10.5" '
        f'font-weight="700" fill="#566072" transform="rotate(-90 18 {divider_y + (H - bottom - divider_y) / 2:.1f})">CL tiers</text>',
    ]

    for pi, (_, title, _, _, _, _) in enumerate(panels):
        panel_x = ml + pi * (plot_w + col_gap)
        out.append(f'<text x="{panel_x + plot_w / 2:.1f}" y="20" text-anchor="middle" '
                   f'font-size="11.5" font-weight="700" fill="#1c2433">{title}</text>')

    for si, (split, label, n_test) in enumerate(SPLITS):
        row_y = header_h + si * (row_h + row_gap)
        plot_top, plot_base = row_y + 10, row_y + 132
        row_label = "v2" if si == 0 else label.lstrip("+")
        out.append(f'<text x="{ml-30}" y="{row_y+63:.1f}" text-anchor="end" font-size="10.5" '
                   f'font-weight="700" fill="#566072">{row_label}</text>')
        out.append(f'<text x="{ml-30}" y="{row_y+78:.1f}" text-anchor="end" font-size="8.5" '
                   f'fill="#9aa3b2">N = {n_test}</text>')

        for pi, (metric, _, vmin, vmax, ticks, higher_better) in enumerate(panels):
            panel_x = ml + pi * (plot_w + col_gap)

            def y(value):
                return plot_base - (value - vmin) / (vmax - vmin) * (plot_base - plot_top)

            for tick in ticks:
                yy = y(tick)
                out.append(f'<line x1="{panel_x:.1f}" y1="{yy:.1f}" x2="{panel_x+plot_w:.1f}" y2="{yy:.1f}" '
                           'stroke="#e6e9ef" stroke-width="1"/>')
                out.append(f'<text x="{panel_x-5:.1f}" y="{yy+3:.1f}" text-anchor="end" font-size="8" '
                           f'fill="#9aa3b2">{tick:.2f}</text>')

            inner = 8
            pitch = (plot_w - inner * 2) / len(methods)
            bw = min(11.0, pitch * 0.58)
            values = []
            for mi, (name, color, flag, *_) in enumerate(methods):
                data = load(name, split)
                value = data.get(metric) if data else None
                if not value or value[0] is None:
                    continue
                mean, sd = value[0], value[1] if value[1] is not None else 0.0
                leaked = name in LEAKED or flag == "leaked"
                values.append((mi, name, color, mean, sd, leaked))

            ranked = [item for item in values if not item[5]]
            best_i = None
            if ranked:
                best_i = (max if higher_better else min)(ranked, key=lambda item: item[3])[0]

            for mi, name, color, mean, sd, leaked in values:
                cx = panel_x + inner + pitch * (mi + 0.5)
                bar_y = max(plot_top, min(plot_base, y(mean)))
                height = max(0.0, plot_base - bar_y)
                opacity = ' opacity="0.38"' if leaked else ""
                stroke = ' stroke="#000" stroke-width="1.5"' if mi == best_i else ""
                out.append(f'<rect x="{cx-bw/2:.1f}" y="{bar_y:.1f}" width="{bw:.1f}" height="{height:.1f}" '
                           f'rx="1.5" fill="{color}"{stroke}{opacity}><title>{name} · {label}: '
                           f'{mean:.3f} ± {sd:.3f}</title></rect>')
                yhi = max(plot_top, min(plot_base, y(mean + sd)))
                ylo = max(plot_top, min(plot_base, y(mean - sd)))
                out.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" '
                           f'stroke="#566072" stroke-width="0.8"{opacity} data-whisker="true" '
                           f'data-ymin="{plot_top:.1f}" data-ymax="{plot_base:.1f}"/>')
                if mi == best_i:
                    label_y = max(plot_top + 9, yhi - 3)
                    out.append(f'<text x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                               f'font-size="8.5" font-weight="700" fill="#1c2433">{mean:.2f}</text>')

            out.append(f'<line x1="{panel_x:.1f}" y1="{plot_base:.1f}" '
                       f'x2="{panel_x+plot_w:.1f}" y2="{plot_base:.1f}" '
                       'stroke="#aeb6c4" stroke-width="1.2"/>')

    out.append(f'<line x1="8" y1="{divider_y:.1f}" x2="{W-mr:.1f}" y2="{divider_y:.1f}" '
               'stroke="#566072" stroke-width="1.6"/>')
    out.append("</svg>")
    return "\n      ".join(out)


def ablation_section():
    """Section 1.3 — CDG-encoder ablation: numbered table + bar chart from ablation_cdg.json."""
    import json
    jp = os.path.join(HERE, "ablation_cdg.json")
    if not os.path.exists(jp):
        return ('<h3 class="subsec-head"><span class="sn">1.3</span> Ablation studies</h3>'
                '<p class="subsec-intro">ablation_cdg.json missing &mdash; run '
                '<code>voxbind/test/ablation_probe.py</code>.</p>')
    rows = sorted(json.load(open(jp)), key=lambda d: d["num"])
    champ = next((d for d in rows if d["num"] == 1), rows[0])
    cr = champ["test_rho"]
    best = max(d["test_rho"] for d in rows)
    COL = {"ref": "#566072", "help": "#2f8f5b", "neutral": "#c98a12", "hurt": "#b3423a"}
    VLAB = {"ref": "baseline", "help": "helps", "neutral": "neutral", "hurt": "hurts"}

    def verdict(d):
        if d["num"] == 1:
            return "ref"
        dd = d["test_rho"] - cr
        return "help" if dd > 0.004 else ("hurt" if dd < -0.004 else "neutral")

    # ---------- bar chart: test rho per ablation number (x-ticks = numbers) ----------
    W, H, ML, MR, MT, MB = 1040, 392, 58, 20, 26, 60
    PW, PH = W - ML - MR, H - MT - MB
    ymin, ymax = 0.44, 0.665
    yof = lambda v: MT + (ymax - v) / (ymax - ymin) * PH
    n = len(rows); slot = PW / n; bw = slot * 0.58
    ybase = yof(ymin)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="CDG ablation">']
    for g in (0.45, 0.50, 0.55, 0.60, 0.65):
        y = yof(g)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eceff3"/>')
        s.append(f'<text x="{ML-8}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="#8a94a6">{g:.2f}</text>')
    ymid = MT + PH / 2
    s.append(f'<text x="15" y="{ymid:.0f}" font-size="11" fill="#566072" text-anchor="middle" '
             f'transform="rotate(-90 15 {ymid:.0f})">test &#961;</text>')
    yc = yof(cr)
    s.append(f'<line x1="{ML}" y1="{yc:.1f}" x2="{ML+PW}" y2="{yc:.1f}" stroke="#566072" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{ML+PW-2}" y="{yc-4:.1f}" text-anchor="end" font-size="9.5" fill="#566072">'
             f'champion &#961; {cr:.3f}</text>')
    for i, d in enumerate(rows):
        cx = ML + slot * (i + 0.5); rho = d["test_rho"]; top = yof(rho); h = ybase - top
        vk = verdict(d); sub = d.get("n_test", 1320) < 1320
        op = "0.5" if sub else "1"
        strk = ' stroke="#1c2433" stroke-width="1.5"' if rho == best else ''
        s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" '
                 f'fill="{COL[vk]}" fill-opacity="{op}"{strk}><title>#{d["num"]} {d["label"]}: '
                 f'test rho {rho:.3f} (n={d.get("n_test","")})</title></rect>')
        s.append(f'<text x="{cx:.1f}" y="{top-4:.1f}" text-anchor="middle" font-size="9.5" fill="#2b3444">{rho:.3f}</text>')
        yv = yof(d["val_rho"])
        s.append(f'<line x1="{cx-bw/2:.1f}" y1="{yv:.1f}" x2="{cx+bw/2:.1f}" y2="{yv:.1f}" '
                 f'stroke="#1c2433" stroke-width="1.3" opacity="0.5"/>')
        star = '*' if sub else ''
        s.append(f'<text x="{cx:.1f}" y="{ybase+17:.1f}" text-anchor="middle" font-size="12" '
                 f'fill="#2b3444" font-weight="700">{d["num"]}{star}</text>')
    s.append(f'<line x1="{ML}" y1="{ybase:.1f}" x2="{ML+PW}" y2="{ybase:.1f}" stroke="#c9d0da"/>')
    ly = H - 14; xcur = ML
    for k in ("ref", "help", "neutral", "hurt"):
        s.append(f'<rect x="{xcur}" y="{ly-8}" width="11" height="11" rx="2" fill="{COL[k]}"/>')
        s.append(f'<text x="{xcur+15}" y="{ly+1}" font-size="10.5" fill="#566072">{VLAB[k]}</text>')
        xcur += 15 + int(len(VLAB[k]) * 6.6) + 16
    s.append(f'<line x1="{xcur}" y1="{ly-3}" x2="{xcur+15}" y2="{ly-3}" stroke="#1c2433" stroke-width="1.3" opacity="0.6"/>')
    s.append(f'<text x="{xcur+20}" y="{ly+1}" font-size="10.5" fill="#566072">val &#961;</text>')
    s.append(f'<text x="{xcur+58}" y="{ly+1}" font-size="10.5" fill="#8a94a6">* n=779 subset</text>')
    s.append('</svg>')
    svg = "".join(s)

    # ---------- table ----------
    thc = "padding:7px 9px;text-align:"
    th = ('<tr style="background:#f6f8fa;color:#7a8699;font-size:11px;text-transform:uppercase;letter-spacing:.04em">'
          f'<th style="{thc}left">#</th><th style="{thc}left">Ablation</th><th style="{thc}left">Group</th>'
          f'<th style="{thc}left">What it tests</th><th style="{thc}right">n</th>'
          f'<th style="{thc}right">val &#961;</th><th style="{thc}right">test r</th>'
          f'<th style="{thc}right">test &#961;</th><th style="{thc}right">RMSE</th>'
          f'<th style="{thc}right">&Delta;&#961;</th><th style="{thc}center">verdict</th></tr>')
    trs = [th]
    for d in rows:
        vk = verdict(d); c = COL[vk]; drho = d["test_rho"] - cr
        dcol = "#2f8f5b" if drho > 0.004 else ("#b3423a" if drho < -0.004 else "#8a94a6")
        dtxt = "&mdash;" if vk == "ref" else f"{drho:+.3f}"
        rowbg = "background:#f3f6f4;" if vk == "ref" else ""
        bold = "font-weight:700;" if d["test_rho"] == best else ""
        num = "font-variant-numeric:tabular-nums;"
        badge = (f'<span style="background:{c};color:#fff;font-size:10px;font-weight:700;'
                 f'padding:1px 8px;border-radius:9px">{VLAB[vk]}</span>')
        trs.append(
            f'<tr style="border-top:1px solid #edf0f4;{rowbg}">'
            f'<td style="padding:6px 9px;color:#7a8699;font-weight:700;font-family:ui-monospace,monospace">{d["num"]}</td>'
            f'<td style="padding:6px 9px;{bold}">{d["label"]}</td>'
            f'<td style="padding:6px 9px;color:#7a8699;font-size:12px">{d["group"]}</td>'
            f'<td style="padding:6px 9px;color:#5b6678;font-size:12.5px">{d["what"]}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}color:#8a94a6">{d.get("n_test","")}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}color:#5b6678">{d["val_rho"]:.3f}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}color:#5b6678">{d["test_r"]:.3f}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}{bold}">{d["test_rho"]:.3f}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}color:#5b6678">{d["test_rmse"]:.3f}</td>'
            f'<td style="padding:6px 9px;text-align:right;{num}color:{dcol};font-weight:600">{dtxt}</td>'
            f'<td style="padding:6px 9px;text-align:center">{badge}</td></tr>')
    table = ('<table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff;'
             'border:1px solid #e3e7ee;border-radius:10px;overflow:hidden">' + "".join(trs) + '</table>')

    intro = ('<ul class="caption-list">'
             '<li>Same frozen mean-pool &rarr; MLP probe; MSE, 5 seeds, <code>lp_edrscc_v2</code>.</li>'
             '<li>Only ensembles #4/#5 exceed the champion, by about +0.005 &#961;.</li>'
             '<li>* #13/#14 use N=779; matched champion &#961; = 0.611.</li></ul>')
    header = '<h3 class="subsec-head"><span class="sn">1.3</span> Ablation studies</h3>'
    return (header + intro
            + '<section class="block"><p class="table-title">Figure 3 &nbsp;&middot;&nbsp; Test &#961; by ablation '
              'number &mdash; champion (dashed) vs attempts</p>'
              '<div class="table-wrap" style="padding:14px 16px 8px">' + svg + '</div></section>'
            + '<section class="block"><p class="table-title">Table 3 &nbsp;&middot;&nbsp; CDG-encoder ablation '
              '&mdash; val &#961; / test r / test &#961; / RMSE (5 seeds, MSE head)</p>'
              '<div class="table-wrap">' + table + '</div></section>')


def build():
    css, table2, sub2, vina_img = extract_from_715()
    table2 = add_atom_column(table2)
    table2 = inject_decompdiff_repro(table2)
    table2 = mark_voxbind_reproduced(table2)
    table2 = strip_repeated_table_units(table2)
    vina_img = denovo_chart.render(table2)
    pdbbind2016_chart = casf_bar_svg("clean")
    pdbbind2016_legend = casf_bar_legend()
    holdout2019_common_chart = holdout2019_common_bar_svg()
    holdout2019_common_legend = holdout2019_common_bar_legend()
    lp_tiers_chart = lp_all_tiers_svg()
    lba_chart = lba_bar_svg()
    lba_legend = lba_bar_legend()
    legend = bar.legend(leaked_names=LEAKED)
    abl_html = ablation_section()          # §1.3 CDG-encoder ablation (reads ablation_cdg.json)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxBind — Results</title>
{css}
<style>
  /* three distinctly-coloured method tags */
  table.results td.col-method .tag.supervised{{background:#eef1f6;color:#5b6678;}}
  table.results td.col-method .tag.pretrained{{background:#e7eefb;color:#38559b;}}
  table.results td.col-method .tag.zeroshot{{background:#f3eefb;color:#6b4b9b;}}
  table.results td.col-method .tag.esm{{background:#ffe3c2;color:#b25a10;font-weight:600;border:1px solid #eec39a;}}
  table.results td.col-method .tag.backbone{{background:#eef0f2;color:#7a8290;font-style:italic;}}
  table.results td.col-method .tag.leaked{{background:#fbe0e0;color:#b03030;font-weight:600;border:1px solid #eeb8b8;}}
  table.results td.col-method .tag.repro{{background:#d8f0e4;color:#1a7a4f;font-weight:600;border:1px solid #a9dcc4;}}
  table.results td.col-method .tag.retrained{{background:#fdf0d8;color:#9a6a15;font-weight:600;border:1px solid #eed9a8;}}
  table.results td.col-modality{{text-align:center;vertical-align:middle;padding:6px 11px;font-size:12px;
    font-weight:600;color:#39424f;line-height:1.4;border-right:1px solid #cfd6df;border-top:1px solid #cfd6df;}}
  table.results th.col-modality{{border-right:1px solid #cfd6df;}}
  table.results td.cat-seq{{background:#faf3e6;}}
  table.results td.cat-3d{{background:#eef1f6;}}
  table.results td.cat-ours{{background:#e2efe7;color:#12633a;}}
  table.results td.cat-leaked{{background:#fbe0e0;color:#b03030;}}
  .lead{{font-size:14px;color:var(--ink-soft);margin:10px 0 0;max-width:1000px;}}
  table.results thead tr.grp th .nsub{{display:block;font-size:9.5px;font-weight:500;letter-spacing:.02em;
    text-transform:none;color:#9aa3b2;margin-top:2px;}}
  .tbd{{color:#b07a17;font-weight:600;font-size:12px;}}
  .na{{color:#c2c8d2;}}
  sup.dg{{color:#b0670f;font-weight:700;cursor:help;margin-left:1px;}}
  table.results .div-block{{border-left:3px solid #1c2433;}}
  table.results thead .div-block{{background:#f4f7fb;}}
  table.results td.metric,table.results th{{padding-left:7px;padding-right:7px;}}
  .metric .val{{font-weight:560;}} .metric .sd{{color:#9aa3b2;font-size:10px;font-weight:400;}}
  /* Table 1 ranking: best = bold (via .best), second = underline */
  table.results td.second .val{{text-decoration:underline;text-underline-offset:2px;text-decoration-thickness:1.5px;}}
  .split-notes{{font-size:13px;color:var(--ink-soft);margin:12px 0 0;max-width:1000px;line-height:1.55;padding-left:20px;}}
  .split-notes li{{margin-bottom:4px;}} .split-notes b{{color:var(--ink);}}
  .split-notes code{{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12.5px;}}
  .caption-list{{max-width:1100px;margin:7px 0 17px;padding-left:20px;color:var(--ink-soft);
    font-size:12.8px;line-height:1.5;}}
  .caption-list li{{margin-bottom:4px;}}
  .caption-list b{{color:var(--ink);font-weight:650;}}
  .table-wrap>.caption-list{{margin:10px 5px 0;}}
  .subsec-head{{font-size:18px;font-weight:680;color:var(--ink);margin:30px 0 4px;padding-bottom:6px;
    border-bottom:2px solid #dfe4ea;}}
  .subsec-head .sn{{display:inline-block;min-width:34px;color:#566072;font-weight:720;}}
  .subsec-intro{{font-size:13.5px;color:var(--ink-soft);margin:8px 0 4px;max-width:1050px;line-height:1.55;}}
</style>
</head><body><div class="page">

  <header class="doc">
    <div class="date">2026 · 07 · 16 — Results</div>
    <h1>VoxBind &mdash; Results</h1>
    <p class="lead">Binding-affinity regression and de novo drug design &mdash; headline tables and charts only.</p>
  </header>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">1</span> Binding-affinity regression</h2>
    <p class="section-intro">Binding-affinity regression across sequence-clustered, leakage-controlled splits.</p>
    <ul class="split-notes">
      <li><b>Data:</b> LP-PDBBind &cap; electron density, RSCC&nbsp;&ge;&nbsp;0.8, Kd/Ki only.</li>
      <li><b>Splits:</b> <code>lp_edrscc_v2</code> &rarr; <code>+CL1</code> &rarr; <code>+CL1+CL2</code> &rarr; <code>+CL1+CL2+CL3</code>.</li>
      <li><b>Protocol:</b> sequence-clustered; each tier keeps the original train/val/test assignment; 3 seeds.</li>
    </ul>
    <p class="subsec-intro"><b>Baseline methods</b></p>
    <ul class="split-notes">
      <li><a href="https://proceedings.mlr.press/v139/satorras21a">EGNN</a> is the ICML 2021 E(<i>n</i>)-equivariant message-passing architecture used as our coordinate-only graph baseline.</li>
      <li><a href="https://arxiv.org/abs/2303.03543">EGNN + TargetDiff</a> augments EGNN with representations from the target-aware equivariant diffusion model published at ICLR 2023.</li>
      <li><a href="https://arxiv.org/abs/2306.01474">GET</a> is the ICML 2024 Generalist Equivariant Transformer, which models multiscale 3D molecular interactions with bilevel E(3)-equivariant attention.</li>
      <li><a href="https://openreview.net/forum?id=A1HhtITVEi">CheapNet</a> is an ICLR 2025 affinity predictor that combines atom-level features, differentiable clustering, and hierarchical cross-attention.</li>
      <li><a href="https://arxiv.org/abs/2311.16160">BindNet</a> is an ICLR 2024 self-supervised protein&ndash;ligand encoder trained through pair-distance prediction and masked-ligand reconstruction.</li>
      <li><a href="https://arxiv.org/abs/2310.07229">ProFSA</a> is an ICLR 2024 frozen pocket encoder pretrained by aligning protein-fragment pseudo-ligands with their surrounding pockets.</li>
      <li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6a45a1b0697ee086bd8bf494cacc6567-Abstract-Conference.html">DSMBind</a> is the NeurIPS 2023 NERE energy model, trained without affinity labels through SE(3) denoising score matching.</li>
      <li><a href="https://openreview.net/forum?id=qH9nrMNTIW">IPNet</a> is the E(3)-equivariant interaction-prior network from the ICLR 2024 IPDiff paper, evaluated here with both its released frozen weights and a from-scratch fit.</li>
      <li><a href="https://www.nature.com/articles/s42004-025-01428-y">AEV-PLIG</a> is a 3D atom-environment-vector affinity model published in <i>Communications Chemistry</i> in 2025.</li>
      <li><a href="https://doi.org/10.1093/bioinformatics/bty593">DeepDTA</a> is a sequence/SMILES convolutional affinity predictor published in <i>Bioinformatics</i> in 2018.</li>
      <li><a href="https://doi.org/10.1093/bioinformatics/btaa880">MolTrans</a> is a substructure-aware sequence/SMILES interaction transformer published in <i>Bioinformatics</i> in 2021.</li>
      <li><a href="https://arxiv.org/abs/2604.23115">HBGSA</a> models hydrogen-bond topology with graph self-attention and is currently an arXiv preprint from 2026 with no conference venue reported.</li>
      <li><a href="https://arxiv.org/abs/2606.03422">HonestAffinity</a> combines frozen ESM-2 features with an explicit pocket-position marker and is currently an arXiv preprint from 2026 with no conference venue reported.</li>
      <li><a href="https://arxiv.org/abs/2606.14217">CPES</a> fuses curvature-derived potential-energy descriptors with geometry-aware message passing and is currently an arXiv preprint from 2026 with no conference venue reported.</li>
      <li><a href="https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf">Nesso-1</a> is a sequence/SMILES affinity model released as a Valence Labs technical report in 2026 with no peer-reviewed venue reported.</li>
    </ul>

    <h3 class="subsec-head"><span class="sn">1.1</span> LP-PDBBind</h3>
    <p class="subsec-intro">In-distribution PDBbind regression on the <code>lp_edrscc_v2</code> test set and its three nested
      no-leak cleaning tiers (+CL1/+CL2/+CL3). Every method is trained &amp; tested within each tier.</p>

    <section class="block">
      <p class="table-title">Figure 1 &nbsp;&middot;&nbsp; LP-PDBBind test metrics across CL cleaning tiers</p>
      <div class="table-wrap" style="padding:16px 18px 10px;">
      {lp_tiers_chart}
      {legend}
      <ul class="caption-list">
        <li>Bars = 3-seed mean; whiskers = &plusmn;1 std.</li>
        <li>The horizontal line separates the original split from CL-filtered tiers.</li>
        <li>Leaked references are faded and excluded from ranking.</li>
      </ul>
      </div>
    </section>

    <section class="block">
      <p class="table-title">Table 1a &nbsp;&middot;&nbsp; Test metrics across cleaning tiers &mdash; mean &plusmn; std (3 seeds; deterministic zero-shot = 1 pass)</p>
      <ul class="caption-list">
        <li><b>CL1 / CL2 / CL3:</b> nested leakage filters; every tier is retrained independently.</li>
        <li><b>Train / val / test:</b> v2 3850/817/1320; +CL1 2721/680/1166; +CL12 2643/659/1149; +CL123 1559/410/733.</li>
        <li>Tables 1a/1b use the same 19-method baseline universe; unavailable runs remain explicit TBA/running placeholders rather than disappearing.</li>
        <li><b>CPES:</b> source code is public but no author checkpoint is released; all cells stay TBA until cohort-specific training.</li>
        <li><span class="tag leaked">leaked</span> rows overlap affinity training and are excluded from ranking.</li>
        <li><b>+corr:</b> same frozen encoder; MSE + Pearson auxiliary loss.</li>
      </ul>
      <div class="table-wrap"><table class="results">
        <thead>
          {affinity_table_head()}
        </thead>
        <tbody>
          {affinity_rows()}
        </tbody>
      </table></div>
    </section>

    <section class="block">
      <p class="table-title">Figure 1b &nbsp;&middot;&nbsp; Generalization across Table 1b cohorts</p>
      <div class="table-wrap" style="padding:16px 18px 10px;">
        {lba_chart}
        {lba_legend}
        <ul class="caption-list">
          <li>Bars = mean; whiskers = &plusmn;1 std; TBA-only methods are omitted.</li>
          <li>Black outline = best non-leaked result within each cohort and metric; leaked references are faded.</li>
        </ul>
      </div>
    </section>

    <section class="block">
      <p class="table-title">Table 1b &nbsp;&middot;&nbsp; Generalization under leakage control &mdash; PLINDER-isolated protein-novelty (LBA) &amp; GEMS CleanSplit (mean &plusmn; std; deterministic zero-shot = 1 pass)</p>
      <ul class="caption-list">
        <li><b>ATOM3D LBA60 / LBA30:</b> released official assignments intersected with
            <code>lp_edrscc_v2</code>, then only downstream valid/test entries similar to the actual
            PLINDER-v2.2 pretraining corpus were removed. PLINDER and downstream train are unchanged.</li>
        <li><b>Isolation rule:</b> MMseqs2 exhaustive search, identity &ge;60% (ID60) or &ge;30% (ID30),
            with &ge;80% query and target coverage (<code>--cov-mode 0</code>). Train/val/test becomes
            2150/20/24 (ID60) and 2078/3/9 (ID30).</li>
        <li><b>Small-cohort warning:</b> especially ID30 (val N=3, test N=9) has very high metric and
            model-selection variance; use these columns as a contamination stress test, not a stable ranking.</li>
        <li><b>CleanSplit:</b> leak-free CASF-2016 subset after ED/RSCC filtering.</li>
        <li><b>Cached re-score coverage:</b> C, C+D+G, +corr, and IPNet-frozen cover every surviving LBA
            complex. Nesso-1 covers 0/9 ID30 and 1/24 ID60 tests, so those cells remain TBA;
            it covers CleanSplit 109/109.</li>
        <li>Test cohorts differ; compare within columns or follow one method across columns.</li>
        <li><b>Density gain is split-dependent:</b> C+D+G&minus;C &Delta;&rho; = +0.048 (base)
            &rarr; &minus;0.011 (PLINDER-clean ID60) &rarr; +0.033 (PLINDER-clean ID30)
            &rarr; +0.012 (CleanSplit leak-free).</li>
      </ul>
      <div class="table-wrap"><table class="results">
        <thead>
          {lba_table_head()}
        </thead>
        <tbody>
          {lba_rows()}
        </tbody>
      </table></div>
    </section>

    <h3 class="subsec-head"><span class="sn">1.2</span> Compared to recent works with holdout data</h3>
    <p class="subsec-intro">Held-out CASF-2016 clean-92 and PDBbind 2019 temporal benchmarks. CASP16 is in Appendix&nbsp;A.</p>

    <section class="block">
      <p class="table-title">Figure 2 &nbsp;&middot;&nbsp; Held-out CASF-2016 and PDBbind 2019 performance</p>
      <div class="table-wrap" style="padding:12px 16px 16px;">
        <p class="figure-cap" style="text-align:left;font-weight:700;color:#1c2433;margin:4px 0 2px;font-size:13px;">CASF-2016 clean-92</p>
        {pdbbind2016_chart}
        {pdbbind2016_legend}
        <p class="figure-cap" style="text-align:left;font-weight:700;color:#1c2433;margin:22px 0 2px;font-size:13px;">PDBbind 2019 holdout &mdash; common ED set (N = {holdout2019_common_n()})</p>
        {holdout2019_common_chart}
        {holdout2019_common_legend}
        <ul class="caption-list">
          <li>2019: Nesso-1 uses partial coverage; DSMBind/IPNet-frozen have no pK-scale RMSE.</li>
        </ul>
      </div>
    </section>

    <section class="block">
      <p class="table-title">Table 2 &nbsp;&middot;&nbsp; Held-out generalization &mdash; <b>CASF-2016</b> (clean-92) &amp; <b>PDBbind 2019 holdout</b></p>
      <ul class="caption-list">
        <li><b>CASF cohort:</b> the <a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00545">official
            CASF-2016 set</a> has 285 complexes (57 targets &times; 5); 214 meet our electron-density/RSCC
            evaluation requirements, and removing 90 v2-train plus 32 v2-val overlaps leaves clean-92.</li>
        <li><b>PLINDER audit:</b> 5/214 evaluable CASF complexes occur in PLINDER-v2
            (<code>1e66</code>, <code>1gpn</code>, <code>2r9w</code>, <code>3ge7</code>, <code>3n7a</code>),
            but none occurs in clean-92.</li>
        <li><b>ProFSA audit:</b> its supervised affinity head excludes clean-92, but its released
            self-supervised train LMDB contains 3/214 evaluable CASF structures
            (<code>2c3i</code>, <code>3u9q</code>, <code>4rfm</code>), including 2/92 clean structures
            (<code>2c3i</code>, <code>4rfm</code>); the result is therefore marked as pretraining overlap,
            not strictly leakage-free.</li>
        <li><b>CASF-2016 clean-92:</b> no winner highlighted; bootstrap 95% CI on &rho; is about &plusmn;0.14.</li>
        <li><b>2019:</b> partial coverage is excluded from ranking; &dagger; reports |r| / |&rho;| for sign-arbitrary energy scores.</li>
      </ul>
      <div class="table-wrap"><table class="results">
        <thead>
          {heldout_table_head()}
        </thead>
        <tbody>
          {heldout_rows()}
        </tbody>
      </table></div>
    </section>

    {abl_html}
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num">2</span> De novo drug design</h2>
    <ul class="caption-list">
      <li><b>Sets:</b> paper rows = 100 pockets; reproduced VoxBind = 79 density pockets; DecompDiff = 98/100 pockets.</li>
      <li><b>ref-informed:</b> reference prior, 25 samples/pocket, Vina exhaustiveness 32; 2 oversized-molecule timeouts removed.</li>
      <li><b>ref-free:</b> subpocket atom-count prior; no reference-ligand input.</li>
      <li><b>Docking check:</b> reproduced reference ligands score &minus;7.44 vs paper &minus;7.26, supporting pipeline calibration.</li>
      <li><b>High aff.:</b> fraction out-docking the reference; <b>Diversity:</b> 1 &minus; mean pairwise Tanimoto.</li>
    </ul>

    <section class="block">
      <p class="table-title">Figure 4 &nbsp;&middot;&nbsp; Vina Score / Min / Dock &mdash; average &amp; median</p>
      <div class="table-wrap" style="padding:16px">{vina_img}</div>
      <ul class="caption-list"><li>AutoDock Vina, kcal/mol; source run: <code>260715</code>.</li></ul>
    </section>

    <section class="block">
      <p class="table-title">Table 4 &nbsp;&middot;&nbsp; De novo generation &mdash; CrossDocked benchmark</p>
      <ul class="caption-list">
        <li>Avg/Med are aggregated over pockets, not individual molecules.</li>
        <li><span class="tag repro">reproduced</span> DecompDiff rows are fully re-scored; unavailable VoxBind outputs remain &mdash;.</li>
        <li>Prior-method values are from the VoxBind paper, Table&nbsp;1.</li>
      </ul>
      <div class="table-wrap">{table2}</div>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num" style="background:#566072">A</span> Appendix &mdash; CASP16 pharmaceutical blind affinity (chymase)</h2>
    <ul class="caption-list">
      <li><b>Benchmark:</b> CASP16 affinity = 140 targets across two proteins (autotaxin&nbsp;123 + chymase&nbsp;17). The
          <b>density-common subset</b> &mdash; complexes with a deposited experimental structure <i>and</i> electron density &mdash;
          is the <b>17 chymase (L1000)</b> only; autotaxin's are embargoed (no public structures/maps, 0). n&nbsp;=&nbsp;17,
          one congeneric series, IC50 &asymp;&nbsp;1&ndash;400&nbsp;nM (pIC50 6.4&ndash;9.0).</li>
      <li><b>Why an appendix (underpowered):</b> a single congeneric series is a within-target potency ranking &mdash; a
          narrower, harder task than the cross-target regression the encoder targets. At n&nbsp;=&nbsp;17 with large seed
          variance no method separation is significant. Density roughly doubles the weak ranking signal
          (C+D+G&nbsp;&rho;&nbsp;0.30 vs C&nbsp;0.15) &mdash; directionally on-thesis, not significant. The leaked sequence baseline
          leads for uninteresting reasons (native IC50 output + SAR memorization).</li>
      <li><b>Setup:</b> C / C+D+G = frozen voxel-ViT encoder + MLP head trained on <code>lp_edrscc_v2</code> train (3 seeds),
          predicting the held-out 17; Nesso-1 = zero-shot cofolding (seq/SMILES).</li>
      <li><b>RMSE not cross-comparable:</b> C / C+D+G heads output pK (Kd/Ki) &rarr; RMSE carries a Kd-vs-IC50 offset;
          Nesso-1 outputs IC50 natively (no offset). Rank on <i>r</i>&thinsp;/&thinsp;&rho;, not RMSE.</li>
      <li><b>Leakage:</b> the 17 test ligands are in no training set; the chymase target is only <i>lightly</i> seen
          (encoder pretrain 4K60/4K69; head-train 1t31/3n7o/4kp0/5yjm &mdash; older, different ligands).
          <span class="tag leaked">leaked</span> Nesso-1 is leaked by design (PDBbind/BindingDB/ChEMBL).</li>
    </ul>

    <section class="block">
      <p class="table-title">Table A1 &nbsp;&middot;&nbsp; CASP16 chymase (L1000) &mdash; held-out affinity (3-seed mean&nbsp;&plusmn;&nbsp;std)</p>
      <div class="table-wrap"><table class="results">
        <thead>
          {casp16_table_head()}
        </thead>
        <tbody>
          {casp16_rows()}
        </tbody>
      </table></div>
    </section>
  </div>

  <div class="doc-section">
    <h2 class="section-head"><span class="sec-num" style="background:#566072">B</span> Appendix &mdash; Data processing: density normalization</h2>
    <ul class="caption-list">
      <li><b>Input:</b> experimental 2F<sub>o</sub>&minus;F<sub>c</sub>, resampled at the ligand pose.</li>
      <li><b>Issue:</b> heavy positive tail with near-zero and negative noise.</li>
      <li><b>Transform:</b> arcsinh soft-squash, then z-score.</li>
    </ul>

    <section class="block">
      <p class="table-title">Normalization recipe</p>
      <p class="table-sub" style="font-size:15px;color:var(--ink);margin-bottom:10px;">
        <code>x&prime; = ( arcsinh(x / s) &minus; &mu;<sub>a</sub> ) / &sigma;<sub>a</sub></code>
        &nbsp;&nbsp;with&nbsp;&nbsp;<code>s = 0.5</code>,&nbsp; <code>&mu;<sub>a</sub> = &minus;0.0154</code>,&nbsp;
        <code>&sigma;<sub>a</sub> = 0.508</code> &nbsp;(PLINDER pooled stats, fit on raw crops).</p>
      <ul class="caption-list">
        <li><code>s=0.5</code> sits above the noise floor; zeros and negatives are retained without clipping.</li>
        <li>2.56M voxels: skew 2.22&rarr;0.92; excess kurtosis 18.8&rarr;1.1; 0.64% remain beyond &plusmn;3&sigma;.</li>
      </ul>
    </section>

    <ul class="caption-list">
      <li><b>Precedent:</b> astronomy's <a href="https://iopscience.iop.org/article/10.1086/301004/pdf">asinh magnitude</a>
        and cytometry's <a href="https://bioconductor.org/packages/devel/bioc/vignettes/CATALYST/inst/doc/preprocessing.html">arcsinh cofactor</a>.</li>
      <li>This is a deliberate cross-domain transform for high-dynamic-range density values.</li>
    </ul>
  </div>

</div></body></html>"""
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT}  ({len(html)} bytes)")
    methods = all_methods()
    done = sum(1 for m in methods for s, _, _ in SPLITS if load(m, s))
    casf = sum(1 for m in methods for which, _, _ in CASF_COLS if load_casf(m, which))
    print(f"  filled LP {done}/{len(methods) * len(SPLITS)} | CASF {casf}/{len(methods) * len(CASF_COLS)}")


if __name__ == "__main__":
    build()
