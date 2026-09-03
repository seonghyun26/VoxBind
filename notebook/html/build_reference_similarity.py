"""build_reference_similarity.py — reference-ligand similarity for the drug-design
table, trimmed to the fingerprints that NeurIPS/ICML/ICLR papers actually report.

WHY THIS EXISTS. The 260820 meeting table (260820_meeting.html §1.2) reported seven
similarity columns — Morgan, Scaffold match, 3D shape, MACCS, AtomPair, RDKit, Dice —
and the script that produced them is gone. Six of the seven measure the same thing;
only two are load-bearing in the literature:

  * ECFP4 / Morgan (radius 2, 2048 bit) Tanimoto  -- Rogers & Hahn, JCIM 2010.
    The de facto standard: MOSES's SNN, GuacaMol's nearest-neighbour ECFP4 similarity,
    and the SBDD line (TargetDiff / DecompDiff / Pocket2Mol and the papers that
    benchmark against them) all report Tanimoto over this fingerprint.
  * Bemis-Murcko scaffold  -- Bemis & Murcko, J. Med. Chem. 1996.
    The scaffold-level counterpart; MOSES's Scaff metric and every "scaffold novelty"
    number in the generative-molecule literature use it.

MACCS (Durant et al. 2002), atom pairs (Carhart et al. 1985), the RDKit/Daylight path
fingerprint and count-Morgan/Dice stay available behind --full, as an appendix
robustness check only. Citations: html/reference_similarity_metrics.md.

One exception worth knowing: the *Diversity* column in our Vina tables is NOT this
script's business. TargetDiff's utils/evaluation/similarity.py computes diversity on
Chem.RDKFingerprint, and DecompDiff/DiffSBDD inherited it, so the published diversity
numbers we sit next to are RDKit-fingerprint numbers. Changing that one to Morgan
would silently break comparability; it is left alone on purpose.

AGGREGATION. Per pocket, compare every generated molecule to that pocket's reference
ligand, take the mean and the max, then macro-average those over pockets. Same
convention the 260820 table used, so old and new numbers are comparable.

WHERE THE MOLECULES COME FROM. samples.sdf, parsed with sanitisation, for every method
— one path for all of them. Each target dir also carries a metrics.json whose cached
SMILES are faster to read, but they are a cache and they do not always agree: for our
own runs the two see the same molecules (7,888 and 7,873), while TargetDiff's
metrics.json was written over a filtered subset (7,287 against the SDF's 7,798). Mixing
the two would put methods on different molecule sets, so --from-metrics exists but is
opt-in and prints a warning.

WHAT IS NOT HERE. AR, Pocket2Mol, DiffSBDD and DecompDiff were sampled on the Blackwell
(sm_120) box — see notebook/html/260903/baseline.html appendix A — and only their
aggregates came back, in 260903/baseline_vina.json. There are no per-molecule SMILES on
this machine, so they cannot be measured here. Run this same script over there; see
"ON ANOTHER SERVER" below.

There is deliberately no Reference row: the crystal ligand's similarity to itself is
1.0 and its scaffold always matches, which says nothing.

USAGE
    /opt/conda/envs/voxbind/bin/python notebook/html/build_reference_similarity.py
    ... --full           also emit the appendix fingerprints (MACCS/AtomPair/RDKit/Dice)
    ... --with-3d        also emit 3D shape Tanimoto (slower; needs conformers)
    ... --own-pockets    score each method on all of its own pockets instead of on the
                         set shared by every method (cross-method comparison then is
                         NOT apples to apples -- the n pockets column tells you)
    ... --from-metrics   read cached metrics.json where present instead of samples.sdf
    ... --methods "Ours · v1" TargetDiff        restrict to a subset

ON ANOTHER SERVER (for the baselines that are not on this box)
    Copy this file over, point METHODS at that machine's sample roots -- any directory
    holding target_*/ subdirs with a samples.sdf and the pocket's reference ligand
    beside it (a *_ref.sdf, or the *_lig_*.sdf next to *_pocket10.pdb) -- and run it
    with an rdkit-bearing python. A method may be given a LIST of roots when its run was
    sharded across GPUs. Then copy the resulting reference_similarity.json back here and
    merge, or just paste the printed rows.

    python build_reference_similarity.py --methods AR Pocket2Mol DiffSBDD DecompDiff

OUTPUTS (all under notebook/html/)
    reference_similarity.json        per-pocket values + macro-averaged summary
    reference_similarity.csv         one row per method (mean / median / max)
    reference_similarity_table.html  HTML fragment, read by results2latex.ipynb
    reference_similarity.tex         the paper table: metrics as rows, methods as columns
    reference_similarity_wrap.tex    same rows as a \resizebox'd wraptable (needs wrapfig)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"
BASE = "/home1/irteam/base_drug"
FUNC = "/home1/irteam/funcbind/artifacts/reproduction/crossdocked/paper_run"

# name -> one sample root, or a list of roots when the run was sharded across GPUs.
# A root is any directory holding target_*/ subdirs. Missing roots are skipped with a
# warning, so an entry can be added before its run lands.
#
# NOT PRESENT ON THIS MACHINE (sampled on the Blackwell box, aggregates only in
# 260903/baseline_vina.json): AR, Pocket2Mol, DiffSBDD, DecompDiff. Uncomment and point
# at that machine's paths when running there.
METHODS = {
    "TargetDiff": f"{BASE}/eval/targetdiff",
    "FuncBind": [f"{FUNC}/gpu{i}/samples" for i in range(4)],
    # "AR":         "<blackwell>/…/ar",
    # "Pocket2Mol": "<blackwell>/…/pocket2mol",
    # "DiffSBDD":   "<blackwell>/…/diffsbdd",
    # "DecompDiff": "<blackwell>/…/decompdiff",
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=0.9}": f"{E}/_vanilla_ep923/samples/full_eval_ep923",
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=1.0}": f"{E}/exp_sig1.0_350ep/samples/full_eval_ep349",
    "Ours": f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
    # Our other two arms, measured 2026-09-03 on the same 79 pockets and then dropped
    # from the paper table (ECFP4 mean / median, scaffold match):
    #   sigma=1.0  0.107 / 0.102 / 1.23%   f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig1.0/samples/full_eval_ep349"
    #   ligmask    0.107 / 0.100 / 1.44%   f"{E}/voxbind_frozenenc_atomblob7_v2p1_ligmask_sig0.9/samples/full_eval_ep349"
}
# plain-text labels for console/csv/json, and the HTML markup the fragment carries.
# results2latex.ipynb turns <sub class="sc"> back into \textsubscript{\scriptsize …},
# which is how the de novo table in the paper already writes sigma.
PLAIN = {
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=0.9}": "VoxBind σ=0.9",
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=1.0}": "VoxBind σ=1.0",
}
HTML_LABEL = {
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=0.9}": 'VoxBind<sub class="sc">σ=0.9</sub>',
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=1.0}": 'VoxBind<sub class="sc">σ=1.0</sub>',
}


def html_label(key: str) -> str:
    return HTML_LABEL.get(key, PLAIN.get(key, key))


_MORGAN = rfg.GetMorganGenerator(radius=2, fpSize=2048)
_ATOMPAIR = rfg.GetAtomPairGenerator(fpSize=2048)
_RDKIT_FP = rfg.GetRDKitFPGenerator(fpSize=2048)


# ── fingerprints ──────────────────────────────────────────────────────────────
def _maccs(mol):
    from rdkit.Chem import MACCSkeys
    return MACCSkeys.GenMACCSKeys(mol)


FPS = {
    "ecfp4": ("ECFP4", lambda m: _MORGAN.GetFingerprint(m), DataStructs.TanimotoSimilarity),
    "maccs": ("MACCS", _maccs, DataStructs.TanimotoSimilarity),
    "atompair": ("AtomPair", lambda m: _ATOMPAIR.GetFingerprint(m), DataStructs.TanimotoSimilarity),
    "rdkit": ("RDKit", lambda m: _RDKIT_FP.GetFingerprint(m), DataStructs.TanimotoSimilarity),
    # Dice is conventionally read on the *unfolded* sparse count Morgan fingerprint —
    # folding it to 2048 bits shifts the numbers (0.222 vs 0.208 here), so keep the
    # sparse form the 260820 table used.
    "dice": ("Dice", lambda m: rdMolDescriptors.GetMorganFingerprint(m, 2), DataStructs.DiceSimilarity),
}
HEADLINE_FPS = ["ecfp4"]
APPENDIX_FPS = ["maccs", "atompair", "rdkit", "dice"]

# Per pocket we take all three; the table shows mean and median, and max stays in the
# json/csv because the 260820 table reported it and old numbers get compared to new.
TABLE_STATS = ("mean", "median")
ALL_STATS = ("mean", "median", "max")
STAT_FN = {"mean": st.mean, "median": st.median, "max": max}


def bm_scaffold(mol):
    """Canonical SMILES of the Bemis-Murcko scaffold, or None when there is none
    (acyclic molecules have an empty scaffold and must not count as a match)."""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaf)


# ── data loading ──────────────────────────────────────────────────────────────
def pocket_key(target_dir: str) -> str | None:
    """Normalised pocket id, so every method's target_NN lines up.

    Ours and FuncBind name the pocket file <UNIPROT>__<pdb>_..._pocket10.pdb; the
    baselines drop the UniProt prefix. Key on the part after the last '__'.
    """
    for f in sorted(os.listdir(target_dir)):
        if f.endswith("_pocket10.pdb"):
            return f[: -len(".pdb")].split("__")[-1]
    return None


def load_from_sdf(target_dir: str):
    """(reference_mol, [generated mols]) read from the SDFs — the default path."""
    ref_mol = None
    for f in sorted(os.listdir(target_dir)):
        if f == "samples.sdf" or not f.endswith(".sdf"):
            continue
        if not (f.endswith("_ref.sdf") or "_lig_" in f):
            continue
        for m in Chem.SDMolSupplier(os.path.join(target_dir, f), sanitize=True):
            if m is not None:
                ref_mol = m
                break
        if ref_mol is not None:
            break
    spath = os.path.join(target_dir, "samples.sdf")
    if ref_mol is None or not os.path.exists(spath):
        return None
    mols = [m for m in Chem.SDMolSupplier(spath, sanitize=True) if m is not None]
    return (ref_mol, mols) if mols else None


def load_from_metrics(target_dir: str):
    """Same, from the cached SMILES in metrics.json. No conformers, so no 3D shape."""
    mpath = os.path.join(target_dir, "metrics.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as fh:
        data = json.load(fh)
    ref = data.get("reference") or {}
    if not isinstance(ref, dict) or not ref.get("smiles"):
        return None
    ref_mol = Chem.MolFromSmiles(ref["smiles"])
    if ref_mol is None:
        return None
    mols = []
    for s in data.get("samples") or []:
        m = Chem.MolFromSmiles(s["smiles"]) if s.get("smiles") else None
        if m is not None:
            mols.append(m)
    return (ref_mol, mols) if mols else None


def shape_tanimoto(mol, ref) -> float | None:
    """RDKit shape Tanimoto in the shared pocket frame, without re-alignment — both
    molecules already live in the same crop coordinates, and aligning them would throw
    away exactly the placement information we want to measure."""
    from rdkit.Chem import rdShapeHelpers
    try:
        return 1.0 - float(rdShapeHelpers.ShapeTanimotoDist(mol, ref))
    except Exception:
        return None


# ── per-method computation ────────────────────────────────────────────────────
def method_values(roots, fp_keys, want_3d: bool, from_metrics: bool):
    """{pocket_key: {...}} over every target dir under every root of one method."""
    out = {}
    for root in roots:
        for t in sorted(d for d in os.listdir(root) if d.startswith("target_")):
            tdir = os.path.join(root, t)
            if not os.path.isdir(tdir):
                continue
            key = pocket_key(tdir)
            if key is None:
                continue
            loaded = (load_from_metrics(tdir) or load_from_sdf(tdir)) if from_metrics \
                else load_from_sdf(tdir)
            if loaded is None:
                continue
            ref_mol, mols = loaded
            rec = {"n_mols": len(mols), "target": t, "root": root}

            for fk in fp_keys:
                _, build, sim = FPS[fk]
                try:
                    ref_fp = build(ref_mol)
                except Exception:
                    continue
                sims = []
                for m in mols:
                    try:
                        sims.append(float(sim(build(m), ref_fp)))
                    except Exception:
                        pass
                if sims:
                    for stat in ALL_STATS:
                        rec[f"{fk}_{stat}"] = STAT_FN[stat](sims)

            ref_scaf = bm_scaffold(ref_mol)
            if ref_scaf is not None:
                rec["scaffold_match"] = sum(1 for m in mols if bm_scaffold(m) == ref_scaf) / len(mols)

            if want_3d:
                shapes = [s for s in (shape_tanimoto(m, ref_mol) for m in mols) if s is not None]
                if shapes:
                    for stat in ALL_STATS:
                        rec[f"shape3d_{stat}"] = STAT_FN[stat](shapes)

            if key in out:
                print(f"      warning: pocket {key} seen twice ({out[key]['root']} and {root});"
                      " keeping the first")
                continue
            out[key] = rec
    return out


def macro(per_pocket, field):
    vals = [r[field] for r in per_pocket.values() if r.get(field) is not None]
    return st.mean(vals) if vals else None


# ── emitters ──────────────────────────────────────────────────────────────────
def metric_groups(fp_keys, want_3d):
    """[(group name, [(field, stat label, formatter), ...])] in table order.

    A group with more than one stat becomes a \\multirow block in LaTeX and a colspan
    header in HTML; a group with a single unnamed stat (Scaffold match) spans the two
    stub columns instead.
    """
    groups = [(FPS[fk][0], [(f"{fk}_{s}", s, "f3") for s in TABLE_STATS]) for fk in fp_keys]
    groups.append(("Scaffold match", [("scaffold_match", "", "pct")]))
    if want_3d:
        groups.append(("3D shape", [(f"shape3d_{s}", s, "f3") for s in TABLE_STATS]))
    return groups


def columns(fp_keys, want_3d):
    """Flattened (field, header, formatter) view of metric_groups, for csv/console."""
    return [(field, f"{name} {stat}".strip(), kind)
            for name, stats in metric_groups(fp_keys, want_3d)
            for field, stat, kind in stats]


def fmt(value, kind):
    if value is None:
        return "—"
    return f"{value:.3f}" if kind == "f3" else f"{100 * value:.2f}%"


def fmt_tex(value, kind):
    if value is None:
        return "--"
    return f"{value:.3f}" if kind == "f3" else f"{100 * value:.2f}\\%"


def write_html(path, summary, groups, own_pockets, n_pockets):
    """Methods as rows, with a two-row grouped header — the shape results.html uses,
    so results2latex.ipynb can transpose it back into the paper's table."""
    top, sub = [], []
    for name, stats in groups:
        if len(stats) == 1 and not stats[0][1]:
            top.append(f'<th rowspan="2">{name}</th>')
        else:
            top.append(f'<th class="grp" colspan="{len(stats)}">{name}</th>')
            sub.extend(f"<th>{stat}</th>" for _, stat, _ in stats)
    rows = []
    for label, vals in summary.items():
        cells = "".join(f"<td>{fmt(vals.get(f), k)}</td>"
                        for _, stats in groups for f, _, k in stats)
        rows.append(f'            <tr><td class="col-method">{html_label(label)}</td>{cells}</tr>')
    scope = ("each method on its own pockets" if own_pockets
             else f"{n_pockets} pockets shared by every method")
    html = f"""<p class="table-title">Table S · Reference-ligand similarity — {scope}</p>
<div class="table-wrap">
    <table class="results">
        <thead>
            <tr><th class="col-method stub" rowspan="2">Method</th>{"".join(top)}</tr>
            <tr>{"".join(sub)}</tr>
        </thead>
        <tbody>
{chr(10).join(rows)}
        </tbody>
    </table>
</div>
"""
    with open(path, "w") as fh:
        fh.write(html)


CAPTION = (r"\textbf{Reference-ligand similarity} on the CrossDocked benchmark. "
           r"Metrics are averaged over pockets.")
LABEL = "tab:result-drug-reference-similarity"


def latex_table(summary, groups, wrap: bool = False) -> str:
    """Metrics as rows, methods as columns.

    Two stub columns: a metric group name and its statistic. A group with several
    statistics (ECFP4 -> mean / median) takes a \multirow name; a single-statistic
    group (Scaffold match) spans both stub columns instead, its name broken over two
    lines with \shortstack so the stub stays narrow. \shortstack is plain LaTeX, so
    this needs only booktabs and multirow.

    wrap=True emits the 260820 layout: a \resizebox'd wraptable for sitting beside the
    body text (needs wrapfig), same rows either way.
    """
    labels = list(summary)
    pad = "        " if wrap else "    "        # \begin{tabular} indent
    row = pad + "    "                          # rows inside it
    body = [
        r"\begin{wraptable}{r}{0.55\textwidth}" if wrap else r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{",
        "        " + CAPTION,
        r"    }",
        rf"    \label{{{LABEL}}}",
    ]
    if wrap:
        body.append(r"    \resizebox{.98\linewidth}{!}{%")
    body += [
        pad + rf"\begin{{tabular}}{{@{{}}ll{'c' * len(labels)}@{{}}}}",
        row + r"\toprule",
        row + " & ".join([r"\multicolumn{2}{@{}l}{\textbf{Method}}"]
                         + [rf"\textbf{{{m}}}" for m in labels]) + r" \\",
        row + r"\midrule",
    ]
    for group_index, (name, stats) in enumerate(groups):
        if group_index:
            body.append(row + r"\addlinespace")
        single = len(stats) == 1 and not stats[0][1]
        for stat_index, (field, stat, kind) in enumerate(stats):
            values = [fmt_tex(summary[m].get(field), kind) for m in labels]
            if single:
                head = (r"\multicolumn{2}{@{}l}{\shortstack[l]{"
                        + r"\\ ".join(name.split(" ", 1)) + "}}")
            elif stat_index == 0:
                head = rf"\multirow{{{len(stats)}}}{{*}}{{{name}}} & {stat}"
            else:
                head = f" & {stat}"
            body.append(row + " & ".join([head, *values]) + r" \\")
    body += [row + r"\bottomrule", pad + r"\end{tabular}"]
    if wrap:
        body.append(r"    }")
    body.append(r"\end{wraptable}" if wrap else r"\end{table}")
    return "\n".join(body)


def write_tex(path, summary, groups, wrap: bool = False):
    with open(path, "w") as fh:
        fh.write(latex_table(summary, groups, wrap) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also report MACCS / AtomPair / RDKit / Dice (appendix only)")
    ap.add_argument("--with-3d", action="store_true",
                    help="also report 3D shape Tanimoto; needs conformers, much slower")
    ap.add_argument("--own-pockets", action="store_true",
                    help="score each method on all of its own pockets rather than on the "
                         "shared set; rows are then NOT directly comparable")
    ap.add_argument("--from-metrics", action="store_true",
                    help="prefer cached metrics.json over samples.sdf (faster, but the "
                         "cache and the SDF disagree for some methods)")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="subset of METHODS keys or their plain labels")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    if args.from_metrics:
        print("  warning: --from-metrics reads a cache that disagrees with the SDFs for "
              "some methods (TargetDiff: 7,287 molecules vs 7,798); rows may sit on "
              "different molecule sets")
    if args.with_3d and args.from_metrics:
        sys.exit("--with-3d needs conformers, which metrics.json does not carry")

    fp_keys = HEADLINE_FPS + (APPENDIX_FPS if args.full else [])
    wanted = METHODS
    if args.methods:
        sel = set(args.methods)
        wanted = {k: v for k, v in METHODS.items() if k in sel or PLAIN.get(k) in sel}
        if not wanted:
            sys.exit(f"no METHODS matched {args.methods}; known: "
                     + ", ".join(PLAIN.get(k, k) for k in METHODS))

    per_method = {}
    for label, roots in wanted.items():
        roots = [roots] if isinstance(roots, str) else list(roots)
        live = [r for r in roots if os.path.isdir(r)]
        if not live:
            print(f"  skip {PLAIN.get(label, label)}: none of {roots} exist")
            continue
        for missing in (r for r in roots if r not in live):
            print(f"  warning: {PLAIN.get(label, label)} shard missing: {missing}")
        print(f"  {PLAIN.get(label, label)} <- " + ", ".join(live))
        per_method[label] = method_values(live, fp_keys, args.with_3d, args.from_metrics)
        print(f"      {len(per_method[label])} pockets, "
              f"{sum(r['n_mols'] for r in per_method[label].values())} molecules")

    if not per_method:
        sys.exit("no methods produced any pockets")

    shared = set.intersection(*(set(v) for v in per_method.values()))
    if not args.own_pockets and not shared:
        sys.exit("the methods share no pockets; re-run with --own-pockets")
    print(f"  {len(shared)} pockets shared by all {len(per_method)} methods"
          + (" (ignored: --own-pockets)" if args.own_pockets else ""))

    fields = [f"{fk}_{stat}" for fk in fp_keys for stat in ALL_STATS] + ["scaffold_match"]
    if args.with_3d:
        fields += [f"shape3d_{stat}" for stat in ALL_STATS]

    summary = {}
    for label, per_pocket in per_method.items():
        sub = per_pocket if args.own_pockets else {k: v for k, v in per_pocket.items() if k in shared}
        summary[label] = {f: macro(sub, f) for f in fields}
        summary[label]["n_pockets"] = len(sub)
        summary[label]["n_mols"] = sum(r["n_mols"] for r in sub.values())

    groups = metric_groups(fp_keys, args.with_3d)
    cols = columns(fp_keys, args.with_3d)
    out = args.out_dir
    with open(os.path.join(out, "reference_similarity.json"), "w") as fh:
        json.dump({
            "own_pockets": args.own_pockets,
            "from_metrics": args.from_metrics,
            "n_shared_pockets": len(shared),
            "shared_pockets": sorted(shared),
            "fingerprints": fp_keys,
            "with_3d": args.with_3d,
            "summary": {PLAIN.get(k, k): v for k, v in summary.items()},
            "per_pocket": {PLAIN.get(k, k): v for k, v in per_method.items()},
        }, fh, indent=2)

    with open(os.path.join(out, "reference_similarity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_pockets", "n_mols"] + [f for f, _, _ in cols])
        for label, vals in summary.items():
            w.writerow([PLAIN.get(label, label), vals["n_pockets"], vals["n_mols"]]
                       + [vals.get(f) for f, _, _ in cols])

    write_html(os.path.join(out, "reference_similarity_table.html"),
               summary, groups, args.own_pockets, len(shared))
    write_tex(os.path.join(out, "reference_similarity.tex"), summary, groups)
    write_tex(os.path.join(out, "reference_similarity_wrap.tex"), summary, groups, wrap=True)

    width = max(len(PLAIN.get(k, k)) for k in summary)
    print()
    print("  " + "method".ljust(width) + "  " + "pockets".rjust(7) + "  " + "n mol.".rjust(8)
          + "  " + "  ".join(h.rjust(14) for _, h, _ in cols))
    for label, vals in summary.items():
        print("  " + PLAIN.get(label, label).ljust(width)
              + "  " + str(vals["n_pockets"]).rjust(7)
              + "  " + str(vals["n_mols"]).rjust(8) + "  "
              + "  ".join(fmt(vals.get(f), k).rjust(14) for f, _, k in cols))
    print()
    print(f"  wrote reference_similarity.{{json,csv,tex}} + _wrap.tex + _table.html in {out}")


if __name__ == "__main__":
    main()
