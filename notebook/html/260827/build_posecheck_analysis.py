#!/usr/bin/env python3
"""build_posecheck_analysis.py — section 2.2 of results_drug_design.html.

Table 4 reports one number per pose metric. This decomposes them, on the same 78 pockets
and straight out of each pocket's ``metrics.json``:

    Table 5  which of PoseBusters' checks actually fail — the headline PB-valid rate turns
             out to be dominated by a single geometric artefact
    Table 6  the strain distribution rather than its median, including the tail mass the
             median hides and the mean is destroyed by
    Table 7  the PoseCheck interaction profile, which Table 4 never reports at all
    Table 8  whether pose quality trades against docking score, before and after
             controlling for molecule size

Every table is also written to CSV next to this script, so the numbers in the page have a
machine-readable source:

    posecheck_pb_checks_78.csv      posecheck_strain_dist_78.csv
    posecheck_interactions_78.csv   posecheck_quality_vs_dock_78.csv

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/build_posecheck_analysis.py
"""
import collections
import csv
import glob
import json
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "results_drug_design.html")
EXPS = "/home1/irteam/VoxBind/voxbind/exps"
V1 = os.path.join(EXPS, "voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350")
VANILLA = os.path.join(EXPS, "_vanilla_ep923/samples/full_eval_ep923")

# label, run root, "ref" = read the pocket's crystal ligand instead of the generated samples
SERIES = [
    ("Reference ligand", VANILLA, "ref"),
    ("TargetDiff", "/home1/irteam/base_drug/eval/targetdiff", "gen"),
    ("VoxBind &sigma;=0.9", VANILLA, "gen"),
    ("Ours &middot; v1", V1, "gen"),
    ("Ours &middot; v2", os.path.join(EXPS, "samples_reference_receptor_ed_ep350"), "gen"),
]
PLAIN = {"VoxBind &sigma;=0.9": "VoxBind sigma=0.9", "Ours &middot; v1": "Ours v1",
         "Ours &middot; v2": "Ours v2"}
RING = "non-aromatic_ring_non-flatness"
INTERACTIONS = ["Hydrophobic", "VdWContact", "HBAcceptor", "HBDonor"]


def pockets():
    """The 78 shared pockets: every pocket Ours v1 covers, minus target_71."""
    return sorted({os.path.basename(d) for d in glob.glob(os.path.join(V1, "target_*"))} - {"target_71"})


def records(root, mode, targets):
    for t in targets:
        path = os.path.join(root, t, "metrics.json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path, encoding="utf-8"))
        for r in ([doc.get("reference")] if mode == "ref" else (doc.get("samples") or [])):
            if isinstance(r, dict):
                yield t, r


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def partial(x, y, z):
    """Spearman of x,y with z partialled out."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


# ── analyses ────────────────────────────────────────────────────────────────
def pb_checks(targets):
    out = {}
    for label, root, mode in SERIES:
        n = valid = ring = valid_excl = ring_only = 0
        fails = collections.Counter()
        for _, r in records(root, mode, targets):
            pb = r.get("posebusters")
            if not (isinstance(pb, dict) and isinstance(pb.get("valid"), bool)):
                continue
            n += 1
            failed = {k for k, v in (pb.get("checks") or {}).items() if v is False}
            fails.update(failed)
            valid += not failed
            ring += RING in failed
            valid_excl += not (failed - {RING})
            ring_only += failed == {RING}
        if n:
            out[label] = dict(n=n, valid=100 * valid / n, ring=100 * ring / n,
                              valid_excl=100 * valid_excl / n, ring_only=100 * ring_only / n,
                              fails={k: 100 * c / n for k, c in fails.items()})
    return out


def strain_dist(targets):
    out = {}
    for label, root, mode in SERIES:
        v = [float(r["posecheck"]["strain"]) for _, r in records(root, mode, targets)
             if isinstance(r.get("posecheck"), dict) and r["posecheck"].get("strain") is not None]
        a = np.array(v)
        out[label] = dict(n=len(a), **{f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)},
                          gt100=100 * float((a > 100).mean()), gt500=100 * float((a > 500).mean()),
                          gt1e4=100 * float((a > 1e4).mean()))
    return out


def interactions(targets):
    out = {}
    for label, root, mode in SERIES:
        counts = collections.defaultdict(list)
        total, size, clash = [], [], []
        for _, r in records(root, mode, targets):
            pc = r.get("posecheck")
            if not (isinstance(pc, dict) and isinstance(pc.get("interactions"), dict)):
                continue
            it = pc["interactions"]
            for k in INTERACTIONS:
                counts[k].append(it.get(k, 0))
            total.append(pc.get("n_interactions") or sum(it.values()))
            size.append(r.get("n_atoms") or 0)
            clash.append(pc.get("clashes") if pc.get("clashes") is not None else np.nan)
        if not total:
            continue
        mean_size = float(np.mean(size))
        out[label] = dict(n=len(total), atoms=mean_size, total=float(np.mean(total)),
                          per10=10 * float(np.mean(total)) / mean_size,
                          clash10=10 * float(np.nanmean(clash)) / mean_size,
                          **{k: float(np.mean(counts[k])) for k in INTERACTIONS})
    return out


def quality_vs_dock(targets):
    out = {}
    for label, root, mode in SERIES:
        if mode == "ref":
            continue
        path = os.path.join(root, "eval_docking_results.json")
        if not os.path.exists(path):
            continue
        dock = {t["target"]: t for t in json.load(open(path, encoding="utf-8"))["per_target"]}
        S, C, D, A = [], [], [], []
        for t in targets:
            if t not in dock:
                continue
            per_mol = collections.defaultdict(list)
            for m in dock[t].get("per_mol") or []:
                if isinstance(m, dict) and m.get("smiles") and m.get("vina_dock") is not None:
                    per_mol[m["smiles"]].append(m["vina_dock"])
            for _, r in records(root, mode, [t]):
                pc, smi = r.get("posecheck") or {}, r.get("smiles")
                # only unambiguous 1-to-1 SMILES matches: metrics rows are the sanitisable
                # subset, so positional joining is wrong for at least one run
                if not isinstance(pc, dict) or len(per_mol.get(smi, [])) != 1:
                    continue
                if pc.get("strain") is None or pc.get("clashes") is None:
                    continue
                S.append(pc["strain"]); C.append(pc["clashes"])
                D.append(per_mol[smi][0]); A.append(r.get("n_atoms") or 0)
        S, C, D, A = map(np.array, (S, C, D, A))
        out[label] = dict(n=len(S), strain=spearman(S, D), strain_p=partial(S, D, A),
                          clash=spearman(C, D), clash_p=partial(C, D, A), atoms=spearman(A, D))
    return out


# ── rendering ───────────────────────────────────────────────────────────────
def write_csv(name, header, rows):
    path = os.path.join(HERE, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def cell(value, div="", cls="", fmt="{:.1f}", sub=None):
    klass = " ".join(x for x in ("metric", div, cls) if x)
    inner = ('<span class="tbd">n/a</span>' if value is None
             else f'<span class="val">{fmt.format(value)}</span>')
    if sub:
        inner += f' <span class="sd">{sub}</span>'
    return f'<td class="{klass}">{inner}</td>'


def method_cell(label, note=None):
    sub = f'<span style="display:block;font-size:11px;color:#7a8699">{note}</span>' if note else ""
    return f'<td class="col-method">{label}{sub}</td>'


def wrap(title, thead, body, note):
    return (f'  <p class="table-title">{title}</p>\n  <div class="table-wrap">\n'
            f'  <table class="results">\n    <thead>\n{thead}\n    </thead>\n'
            f'    <tbody>\n{body}\n    </tbody>\n  </table>\n  </div>\n'
            f'  <p class="meta">{note}</p>\n')


def splice(html, key, markup):
    a, b = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
    if a not in html:
        raise SystemExit(f"{key}: markers not found in {DOC}")
    return html[:html.index(a) + len(a)] + "\n" + markup + "  " + html[html.index(b):]


def table5(pb):
    cols = [l for l, *_ in SERIES if l in pb]
    thead = ('      <tr class="grp"><th class="col-method">PoseBusters check</th>'
             + "".join(f'<th class="{"div-major" if i == 0 else "div-minor"}">{c}'
                       f'<span class="nsub">n&nbsp;=&nbsp;{pb[c]["n"]:,}</span></th>'
                       for i, c in enumerate(cols)) + "</tr>")
    n = len(cols) + 1
    rows = [f'      <tr class="grp-band"><td colspan="{n}">Summary</td></tr>']

    def line(name, key, note=None, best_high=True):
        vals = [pb[c][key] for c in cols]
        star = max(vals) if best_high else min(vals)
        cells = "".join(cell(v, "div-major" if i == 0 else "div-minor",
                             "best" if v == star and len(set(vals)) > 1 else "", "{:.1f}%")
                        for i, v in enumerate(vals))
        return f"      <tr>{method_cell(name, note)}{cells}</tr>"

    rows.append(line("PB-valid", "valid", "all checks pass"))
    rows.append(line("PB-valid, ring flatness excluded", "valid_excl",
                     "every other check passes"))
    rows.append(line("fails <i>only</i> ring flatness", "ring_only",
                     "the gap between the two rows above", best_high=False))
    rows.append(f'      <tr class="grp-band"><td colspan="{n}">Failure rate by check &middot; '
                f'% of molecules &middot; checks that never fail are omitted</td></tr>')
    names = sorted({k for c in cols for k in pb[c]["fails"]},
                   key=lambda k: -max(pb[c]["fails"].get(k, 0) for c in cols))
    for k in names:
        vals = [pb[c]["fails"].get(k, 0.0) for c in cols]
        cells = "".join(cell(v, "div-major" if i == 0 else "div-minor",
                             "worst" if v == max(vals) and max(vals) > 5 else "", "{:.2f}%")
                        for i, v in enumerate(vals))
        pretty = k.replace("_", " ")
        rows.append(f"      <tr>{method_cell(pretty)}{cells}</tr>")
    note = ("PoseBusters was never run for TargetDiff or Ours&nbsp;v2, and where it did run only about "
            "half the molecules carry a verdict &mdash; these rates are over that subset, which is not "
            "guaranteed unbiased. Percentages are of molecules, and a molecule can fail several checks, "
            "so the failure column does not sum to 100&nbsp;&minus;&nbsp;PB-valid.")
    return wrap("Table 5 &nbsp;&middot;&nbsp; PoseBusters &mdash; which checks actually fail",
                thead, "\n".join(rows), note)


def table6(dist):
    thead = ('      <tr class="grp"><th class="col-method" rowspan="2">Method</th>'
             '<th class="div-major" rowspan="2">n</th>'
             '<th class="div-major" colspan="5">Strain energy percentile</th>'
             '<th class="div-major" colspan="3">Share of molecules above</th></tr>\n'
             '      <tr class="sub"><th class="div-major">p10</th><th>p25</th>'
             '<th>p50<span class="nsub">median</span></th><th>p75</th><th>p90</th>'
             '<th class="div-major">100</th><th>500</th><th>10<sup>4</sup></th></tr>')
    rows = []
    for label, _, mode in SERIES:
        d = dist.get(label)
        if not d:
            continue
        if label == "TargetDiff":
            rows.append('      <tr class="grp-band"><td colspan="10">Generated</td></tr>')
        cells = (cell(d["n"], "div-major", "", "{:,.0f}")
                 + "".join(cell(d[f"p{p}"], "div-major" if p == 10 else "", "", "{:,.1f}")
                           for p in (10, 25, 50, 75, 90))
                 + "".join(cell(d[k], "div-major" if k == "gt100" else "", "", "{:.1f}%")
                           for k in ("gt100", "gt500", "gt1e4")))
        rows.append(f"      <tr>{method_cell(label)}{cells}</tr>")
    note = ("The mean is not shown because it is not usable: a handful of poses where UFF relaxation "
            "fails carry it into the millions. The percentiles say the same thing the median does, "
            "consistently across the whole distribution, and the last three columns are the tail the "
            "median hides.")
    return wrap("Table 6 &nbsp;&middot;&nbsp; Strain energy &mdash; the distribution, not just the median",
                thead, "\n".join(rows), note)


def table7(prof):
    thead = ('      <tr class="grp"><th class="col-method" rowspan="2">Method</th>'
             '<th class="div-major" rowspan="2">Heavy<span class="nsub">atoms, mean</span></th>'
             '<th class="div-major" colspan="2">All interactions</th>'
             '<th class="div-major" colspan="4">Mean count per molecule</th>'
             '<th class="div-major" rowspan="2">Steric clashes<span class="nsub">per 10 atoms</span></th></tr>\n'
             '      <tr class="sub"><th class="div-major">per molecule</th><th>per 10 atoms</th>'
             '<th class="div-major">Hydrophobic</th><th>VdW contact</th>'
             '<th>HB acceptor</th><th>HB donor</th></tr>')
    ref = prof.get("Reference ligand")
    rows = []
    for label, _, _ in SERIES:
        d = prof.get(label)
        if not d:
            continue
        if label == "TargetDiff":
            rows.append('      <tr class="grp-band"><td colspan="9">Generated</td></tr>')
        cells = (cell(d["atoms"], "div-major", "", "{:.1f}")
                 + cell(d["total"], "div-major", "", "{:.2f}")
                 + cell(d["per10"], "", "", "{:.2f}")
                 + "".join(cell(d[k], "div-major" if k == INTERACTIONS[0] else "", "", "{:.2f}")
                           for k in INTERACTIONS)
                 + cell(d["clash10"], "div-major", "", "{:.2f}"))
        rows.append(f"      <tr>{method_cell(label)}{cells}</tr>")
    note = ("PoseCheck's third output, which Table&nbsp;4 does not report. Counts are prolif interaction "
            "fingerprints, averaged over molecules. The reference row is 78 crystal ligands, one per "
            "pocket, so it is a target profile rather than a competitor.")
    return wrap("Table 7 &nbsp;&middot;&nbsp; Protein&ndash;ligand interaction profile", thead,
                "\n".join(rows), note)


def table8(corr):
    thead = ('      <tr class="grp"><th class="col-method" rowspan="2">Method</th>'
             '<th class="div-major" rowspan="2">n<span class="nsub">molecules joined</span></th>'
             '<th class="div-major" colspan="2">Strain vs Vina Dock</th>'
             '<th class="div-major" colspan="2">Clashes vs Vina Dock</th>'
             '<th class="div-major" rowspan="2">Heavy atoms<br>vs Vina Dock</th></tr>\n'
             '      <tr class="sub"><th class="div-major">Spearman &rho;</th>'
             '<th>&rho; | heavy atoms</th>'
             '<th class="div-major">Spearman &rho;</th><th>&rho; | heavy atoms</th></tr>')
    rows = []
    for label, _, _ in SERIES:
        d = corr.get(label)
        if not d:
            continue
        cells = (cell(d["n"], "div-major", "", "{:,.0f}")
                 + cell(d["strain"], "div-major", "", "{:+.3f}")
                 + cell(d["strain_p"], "", "", "{:+.3f}")
                 + cell(d["clash"], "div-major", "", "{:+.3f}")
                 + cell(d["clash_p"], "", "", "{:+.3f}")
                 + cell(d["atoms"], "div-major", "", "{:+.3f}"))
        rows.append(f"      <tr>{method_cell(label)}{cells}</tr>")
    note = ("Vina Dock is negative-is-better, so a <b>negative</b> &rho; means more strain goes with "
            "<i>stronger</i> docking and a <b>positive</b> &rho; means it goes with weaker docking. "
            "&rho;&nbsp;|&nbsp;heavy atoms is the partial Spearman with molecule size removed. Molecules "
            "are joined to their docking score by SMILES, keeping only unambiguous one-to-one matches, "
            "which is why n is below the molecule counts in Table&nbsp;4.")
    return wrap("Table 8 &nbsp;&middot;&nbsp; Does pose quality trade against docking score?", thead,
                "\n".join(rows), note)


def main():
    targets = pockets()
    pb, dist, prof, corr = (pb_checks(targets), strain_dist(targets),
                            interactions(targets), quality_vs_dock(targets))

    write_csv("posecheck_pb_checks_78.csv",
              ["model", "n", "pb_valid_pct", "ring_flatness_fail_pct", "valid_excl_ring_pct",
               "ring_only_fail_pct", "check", "check_fail_pct"],
              [[PLAIN.get(m, m), d["n"], f'{d["valid"]:.2f}', f'{d["ring"]:.2f}',
                f'{d["valid_excl"]:.2f}', f'{d["ring_only"]:.2f}', k, f"{v:.3f}"]
               for m, d in pb.items() for k, v in sorted(d["fails"].items(), key=lambda x: -x[1])])
    write_csv("posecheck_strain_dist_78.csv",
              ["model", "n", "p10", "p25", "p50", "p75", "p90", "pct_gt_100", "pct_gt_500", "pct_gt_1e4"],
              [[PLAIN.get(m, m), d["n"]] + [f'{d[f"p{p}"]:.2f}' for p in (10, 25, 50, 75, 90)]
               + [f'{d[k]:.2f}' for k in ("gt100", "gt500", "gt1e4")] for m, d in dist.items()])
    write_csv("posecheck_interactions_78.csv",
              ["model", "n", "mean_heavy_atoms", "interactions_per_mol", "interactions_per_10_atoms"]
              + [k.lower() for k in INTERACTIONS] + ["clashes_per_10_atoms"],
              [[PLAIN.get(m, m), d["n"], f'{d["atoms"]:.2f}', f'{d["total"]:.3f}', f'{d["per10"]:.3f}']
               + [f'{d[k]:.3f}' for k in INTERACTIONS] + [f'{d["clash10"]:.3f}'] for m, d in prof.items()])
    write_csv("posecheck_quality_vs_dock_78.csv",
              ["model", "n", "rho_strain_dock", "rho_strain_dock_partial_atoms",
               "rho_clash_dock", "rho_clash_dock_partial_atoms", "rho_atoms_dock"],
              [[PLAIN.get(m, m), d["n"], f'{d["strain"]:.4f}', f'{d["strain_p"]:.4f}',
                f'{d["clash"]:.4f}', f'{d["clash_p"]:.4f}', f'{d["atoms"]:.4f}']
               for m, d in corr.items()])

    html = open(DOC, encoding="utf-8").read()
    html = splice(html, "POSECHECK-ANALYSIS",
                  table5(pb) + "\n" + table6(dist) + "\n" + table7(prof) + "\n" + table8(corr))
    open(DOC, "w", encoding="utf-8").write(html)
    print(f"4 tables spliced into {DOC}; CSVs written to {HERE}")
    for m, d in pb.items():
        print(f"  PB {PLAIN.get(m, m):18s} valid {d['valid']:5.1f}%  "
              f"excl-ring {d['valid_excl']:5.1f}%  ring-only {d['ring_only']:5.1f}%")
    for m, d in corr.items():
        print(f"  rho {PLAIN.get(m, m):18s} strain/dock {d['strain']:+.3f} -> partial {d['strain_p']:+.3f}")


if __name__ == "__main__":
    main()
