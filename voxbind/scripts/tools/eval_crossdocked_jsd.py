#!/usr/bin/env python3
"""eval_crossdocked_jsd.py — how closely generated molecules match the CrossDocked test
set's geometry and composition: the TargetDiff / VoxBind distribution metrics.

    python voxbind/scripts/tools/eval_crossdocked_jsd.py \
        --run AR=<root> --run CoDE=<root> ... --out crossdocked_jsd.json [--selfcheck]

Each <root> holds target_*/samples.sdf. `voxbind/scripts/89_eval_crossdocked_jsd.sh` runs
it over the eight arms figures/draw.py draws; `draw.py jsd` then plots the JSON.

WHAT IS MEASURED — each one is the published definition, checked against the paper:

  bond JSD      per bond type (C-C C=C C-N C=N C-O C=O C:C C:N), the lengths binned
                every 0.005 Å over [1.1, 1.7) with an under- and an overflow bin
                (TargetDiff `eval_bond_length_config.DISTANCE_BINS`). TargetDiff Table 3 /
                VoxBind Table 2.
  pair JSD      heavy-atom pair distances within a molecule: all pairs < 12 Å and C-C
                pairs < 2 Å, 100 bins each (TargetDiff Fig. 2).
  atom-type JSD heavy-atom element shares over C N O F P S Cl (TargetDiff
                `eval_atom_type`); anything else is recorded as `other` and left out.
  ring sizes    share of RINGS of each size, 3-9 (TargetDiff Table 2). It is a share of
                rings, not of molecules: every column of the published table sums to 100,
                which `print_ring_ratio` in evaluate_diffusion.py (share of molecules
                holding at least one ring of that size) cannot do. Rings are RDKit's
                `GetRingInfo().AtomRings()`, as in TargetDiff's `scoring_func.get_chem`.
  Fig. 10       ring sizes over all rings, rings per molecule, and the fraction of a
                molecule's heavy atoms that are aromatic (VoxBind Fig. 10).

THE REFERENCE IS THE 100 CROSSDOCKED TEST LIGANDS, not the histograms TargetDiff ships in
`eval_bond_length_config.EMPIRICAL_DISTRIBUTIONS`. Those were built from the TRAINING set
(~1.29M aromatic C:C bonds) and do not give the published numbers. Scoring TargetDiff's
own 9,878 molecules against the 100 test ligands does (C-C .370 / C=N .548 / C:N .235 for
the published .369 / .550 / .235), and against the shipped histograms it does not
(.299 / .165 / .131). `--selfcheck` re-runs that comparison, so the claim stays checked.

CAVEAT: the reference is small (716 C-C bonds but only 40 C=C and 16 C=N), and JSD against
a sparse histogram is inflated by the sparsity alone. The C=C and C=N columns therefore
mostly measure the reference's size; read them as a ranking, never as a distance to zero.

`jensenshannon` is scipy's, which returns the JS DISTANCE (the square root of the
divergence, natural log). Both papers print that number and call it a divergence; it is
kept so the columns stay on the published scale.

THE MOLECULE SET: every entry of samples.sdf that RDKit sanitises and that is a single
connected component -- TargetDiff scores "complete" molecules only, and
notebook/webapp/metrics.py drops disconnected ones before every other metric, so the
figures for one run all stand on the same molecules. Bond orders are what RDKit perceives
on reading the SDF; TargetDiff's own evaluator read them off its reconstruction instead,
which is why the self-check agrees to ~0.005 rather than exactly.
"""
import argparse
import collections
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

REPO = Path(__file__).resolve().parents[3]
REFERENCE_DIR = REPO / "targetdiff" / "data" / "test_set" / "test_set"
P79_JSON = REPO / "voxbind" / "exps" / "frozenenc_probes" / "p79_targets.json"
BASEDRUG = Path(os.environ.get("VOXBIND_BASEDRUG", REPO.parent / "base_drug"))

# TargetDiff utils/evaluation/eval_bond_length_config.py, copied rather than imported:
# importing it pulls TargetDiff's `utils` package (torch and all) onto sys.path.
BOND_BINS = np.arange(1.1, 1.7, 0.005)[:-1]
PAIR_BINS = {"All_12A": np.linspace(0, 12, 100), "CC_2A": np.linspace(0, 2, 100)}
PAIR_MAX = {"All_12A": 12.0, "CC_2A": 2.0}
# (Z1, Z2, order) with Z1 <= Z2; order 1 single, 2 double, 4 aromatic (TargetDiff
# utils.data.BOND_TYPES). Column order of the published tables.
BOND_TYPES = {"C-C": (6, 6, 1), "C=C": (6, 6, 2), "C-N": (6, 7, 1), "C=N": (6, 7, 2),
              "C-O": (6, 8, 1), "C=O": (6, 8, 2), "C:C": (6, 6, 4), "C:N": (6, 7, 4)}
ATOM_TYPES = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
RING_TABLE_SIZES = tuple(range(3, 10))
AROM_EDGES = np.linspace(0, 1, 21)

# The published numbers, for the self-check and as context beside a re-evaluation.
# Bond JSD: VoxBind arXiv:2405.03961 Table 2 (its TargetDiff/AR/Pocket2Mol rows are
# TargetDiff's own Table 3). Ring shares: TargetDiff arXiv:2303.03543 Table 2, percent.
PUBLISHED_BOND_JSD = {
    "AR":            [.609, .620, .474, .635, .492, .558, .451, .552],
    "Pocket2Mol":    [.496, .561, .416, .629, .454, .516, .416, .487],
    "TargetDiff":    [.369, .505, .363, .550, .421, .461, .263, .235],
    "DecompDiff":    [.359, .537, .344, .584, .376, .374, .251, .269],
    "VoxBind σ=0.9": [.372, .528, .351, .528, .400, .326, .215, .186],
    "VoxBind σ=1.0": [.357, .533, .354, .418, .354, .335, .210, .191],
}
PUBLISHED_RING_PCT = {
    "Reference ligand": [1.7, 0.0, 30.2, 67.4, 0.7, 0.0, 0.0],
    "liGAN":            [28.1, 15.7, 29.8, 22.7, 2.6, 0.8, 0.3],
    "AR":               [29.9, 0.0, 16.0, 51.2, 1.7, 0.7, 0.5],
    "Pocket2Mol":       [0.1, 0.0, 16.4, 80.4, 2.6, 0.3, 0.1],
    "TargetDiff":       [0.0, 2.8, 30.8, 50.7, 12.1, 2.7, 0.9],
}
# How far the self-check may sit from the published TargetDiff row. The residual is the
# SDF round trip (see the docstring), measured at 0.006 JSD and 1.1 ring-share points.
SELFCHECK_TOL_JSD, SELFCHECK_TOL_RING = 0.015, 2.0


def profile_sdf(path):
    """Additive counts for one SDF: every histogram is a count vector, so a run's profile
    is the sum over its files and no per-molecule list ever crosses a process boundary."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    p = {
        "n_entries": 0, "n_unparseable": 0, "n_disconnected": 0, "n_mols": 0,
        "bond": {k: np.zeros(len(BOND_BINS) + 1, np.int64) for k in BOND_TYPES},
        "pair": {k: np.zeros(len(b) + 1, np.int64) for k, b in PAIR_BINS.items()},
        "atoms": collections.Counter(), "ring_sizes": collections.Counter(),
        "n_rings": collections.Counter(), "arom": np.zeros(len(AROM_EDGES) - 1, np.int64),
    }
    by_type = {v: k for k, v in BOND_TYPES.items()}
    order_of = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.AROMATIC: 4}
    for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
        p["n_entries"] += 1
        if mol is None:
            p["n_unparseable"] += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            p["n_disconnected"] += 1
            continue
        p["n_mols"] += 1
        pos = mol.GetConformer().GetPositions()

        lengths = collections.defaultdict(list)
        for b in mol.GetBonds():
            order = order_of.get(b.GetBondType())
            if order is None:
                continue
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            z1, z2 = sorted((mol.GetAtomWithIdx(i).GetAtomicNum(),
                             mol.GetAtomWithIdx(j).GetAtomicNum()))
            name = by_type.get((z1, z2, order))
            if name is not None:
                lengths[name].append(np.linalg.norm(pos[i] - pos[j]))
        for name, v in lengths.items():
            p["bond"][name] += np.bincount(np.searchsorted(BOND_BINS, v),
                                           minlength=len(BOND_BINS) + 1)

        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        z = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy])
        if len(heavy) > 1:
            hp = pos[heavy]
            iu = np.triu_indices(len(heavy), 1)
            d = np.linalg.norm(hp[iu[0]] - hp[iu[1]], axis=1)
            cc = (z[iu[0]] == 6) & (z[iu[1]] == 6)
            for key, sel in (("All_12A", d < PAIR_MAX["All_12A"]),
                             ("CC_2A", cc & (d < PAIR_MAX["CC_2A"]))):
                bins = PAIR_BINS[key]
                p["pair"][key] += np.bincount(np.searchsorted(bins, d[sel]),
                                              minlength=len(bins) + 1)

        p["atoms"].update(z.tolist())
        rings = mol.GetRingInfo().AtomRings()
        p["n_rings"][len(rings)] += 1
        p["ring_sizes"].update(len(r) for r in rings)
        if heavy:
            frac = sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in heavy) / len(heavy)
            p["arom"] += np.histogram([frac], bins=AROM_EDGES)[0]
    return p


def merge(profiles):
    out = None
    for p in profiles:
        if out is None:
            out = p
            continue
        for k in ("n_entries", "n_unparseable", "n_disconnected", "n_mols"):
            out[k] += p[k]
        for k in ("bond", "pair"):
            for name in out[k]:
                out[k][name] += p[k][name]
        for k in ("atoms", "ring_sizes", "n_rings"):
            out[k].update(p[k])
        out["arom"] += p["arom"]
    return out


def _jsd(ref_counts, gen_counts):
    """None where either side has nothing to compare -- a missing bond type is not 0."""
    if ref_counts.sum() == 0 or gen_counts.sum() == 0:
        return None
    return float(jensenshannon(ref_counts / ref_counts.sum(), gen_counts / gen_counts.sum()))


def ring_share(ring_sizes):
    """Percent of the size-3-9 rings at each size -- the published table's denominator, so
    the columns sum to 100 -- plus, apart from it, the share of ALL rings at 10 or more.

    The denominator matters: 2.0% of the test ligands' rings are 10+ macrocycles, and
    dividing by all rings moves the reference's 6-ring share from the published 67.4 to
    66.0. `10+ of all` is kept so an arm that builds macrocycles cannot hide outside the
    table."""
    table = sum(ring_sizes.get(s, 0) for s in RING_TABLE_SIZES)
    total = sum(ring_sizes.values())
    pct = {str(s): (100 * ring_sizes.get(s, 0) / table if table else 0.0)
           for s in RING_TABLE_SIZES}
    pct["10+ of all"] = (100 * sum(v for s, v in ring_sizes.items() if s >= 10) / total
                         if total else 0.0)
    return pct


def summarise(p, ref=None):
    """A profile as JSON: the counts behind every figure, and, given the reference, the
    JSDs against it."""
    atom_total = sum(p["atoms"].values())
    listed = np.array([p["atoms"].get(z, 0) for z in ATOM_TYPES], float)
    out = {
        "n_entries": p["n_entries"], "n_unparseable": p["n_unparseable"],
        "n_disconnected": p["n_disconnected"], "n_mols": p["n_mols"],
        "bond_counts": {k: v.tolist() for k, v in p["bond"].items()},
        "bond_n": {k: int(v.sum()) for k, v in p["bond"].items()},
        "pair_counts": {k: v.tolist() for k, v in p["pair"].items()},
        "pair_n": {k: int(v.sum()) for k, v in p["pair"].items()},
        "atom_counts": {ATOM_TYPES.get(z, str(z)): int(n) for z, n in sorted(p["atoms"].items())},
        "atom_frac": {s: (float(listed[i] / listed.sum()) if listed.sum() else 0.0)
                      for i, s in enumerate(ATOM_TYPES.values())},
        "atom_other_frac": (1 - float(listed.sum()) / atom_total) if atom_total else 0.0,
        "ring_size_counts": {str(s): int(n) for s, n in sorted(p["ring_sizes"].items())},
        "ring_size_pct": ring_share(p["ring_sizes"]),
        "n_rings_counts": {str(s): int(n) for s, n in sorted(p["n_rings"].items())},
        "arom_frac_counts": p["arom"].tolist(),
        "mean_n_rings": (sum(s * n for s, n in p["n_rings"].items()) / p["n_mols"]
                         if p["n_mols"] else None),
    }
    if ref is not None:
        bond = {k: _jsd(ref["bond"][k], p["bond"][k]) for k in BOND_TYPES}
        have = [v for v in bond.values() if v is not None]
        ref_listed = np.array([ref["atoms"].get(z, 0) for z in ATOM_TYPES], float)
        out["jsd"] = {
            "bond": bond,
            "bond_mean": float(np.mean(have)) if len(have) == len(bond) else None,
            "pair": {k: _jsd(ref["pair"][k], p["pair"][k]) for k in PAIR_BINS},
            "atom_type": _jsd(ref_listed, listed),
        }
    return out


def target_sdfs(root, targets):
    missing = [t for t in targets if not os.path.exists(os.path.join(root, t, "samples.sdf"))]
    if missing:
        raise SystemExit(f"{root}: no samples.sdf for {len(missing)} of {len(targets)} "
                         f"pockets (first: {missing[0]})")
    return [os.path.join(root, t, "samples.sdf") for t in targets]


def run_profiles(jobs, workers):
    """{name: merged profile} for {name: [sdf, ...]}, one pool over every file."""
    flat = [(name, f) for name, files in jobs.items() for f in files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(profile_sdf, [f for _, f in flat], chunksize=4))
    grouped = collections.defaultdict(list)
    for (name, _), prof in zip(flat, results):
        grouped[name].append(prof)
    return {name: merge(grouped[name]) for name in jobs}


def print_tables(reference, arms):
    names = list(BOND_TYPES)
    print(f"\n  bond-distance JSD vs the {reference['n_mols']} CrossDocked test ligands")
    print(f"  {'':18s}" + "".join(f"{n:>7s}" for n in names) + f"{'mean':>7s}")
    for lab, a in arms.items():
        j = a["jsd"]["bond"]
        cells = "".join(f"{v:7.3f}" if v is not None else f"{'—':>7s}" for v in j.values())
        mean = a["jsd"]["bond_mean"]
        print(f"  {lab:18s}{cells}{mean:7.3f}" if mean is not None else f"  {lab:18s}{cells}")
    print(f"  {'(reference bonds)':18s}" + "".join(f"{reference['bond_n'][n]:7d}" for n in names))

    print(f"\n  {'':18s}{'mols':>7s}{'discon':>7s}{'All<12Å':>9s}{'C-C<2Å':>8s}"
          f"{'atom':>7s}{'rings':>7s}")
    for lab, a in arms.items():
        j = a["jsd"]
        print(f"  {lab:18s}{a['n_mols']:7d}{a['n_disconnected']:7d}"
              f"{j['pair']['All_12A']:9.3f}{j['pair']['CC_2A']:8.3f}{j['atom_type']:7.3f}"
              f"{a['mean_n_rings']:7.2f}")

    sizes = [str(s) for s in RING_TABLE_SIZES] + ["10+ of all"]
    print("\n  % of size-3-9 rings by size (TargetDiff Table 2); last column % of all rings")
    print(f"  {'':18s}" + "".join(f"{s:>6s}" for s in sizes))
    for lab, a in [("Reference ligand", reference)] + list(arms.items()):
        print(f"  {lab:18s}" + "".join(f"{a['ring_size_pct'][s]:6.1f}" for s in sizes))


def selfcheck(reference_profile, workers):
    """TargetDiff's own samples over all 100 pockets against the published row. Returns
    the report; the caller decides whether a miss fails the run."""
    root = BASEDRUG / "eval" / "targetdiff"
    files = sorted(glob.glob(str(root / "target_*" / "samples.sdf")))
    if len(files) != 100:
        raise SystemExit(f"selfcheck: expected 100 TargetDiff pockets under {root}, "
                         f"found {len(files)}")
    td = summarise(run_profiles({"TargetDiff": files}, workers)["TargetDiff"],
                   reference_profile)
    ref = summarise(reference_profile)
    got = [td["jsd"]["bond"][k] for k in BOND_TYPES]
    d_jsd = max(abs(a - b) for a, b in zip(got, PUBLISHED_BOND_JSD["TargetDiff"]))
    sizes = [str(s) for s in RING_TABLE_SIZES]
    d_ring = max(max(abs(td["ring_size_pct"][s] - v)
                     for s, v in zip(sizes, PUBLISHED_RING_PCT["TargetDiff"])),
                 max(abs(ref["ring_size_pct"][s] - v)
                     for s, v in zip(sizes, PUBLISHED_RING_PCT["Reference ligand"])))
    print("\n  SELFCHECK — TargetDiff, 100 pockets, against the published tables")
    print("  bond JSD   ours " + " ".join(f"{v:.3f}" for v in got))
    print("        published " + " ".join(f"{v:.3f}" for v in PUBLISHED_BOND_JSD["TargetDiff"]))
    print("  ring %     ours " + " ".join(f"{td['ring_size_pct'][s]:5.1f}" for s in sizes))
    print("        published " + " ".join(f"{v:5.1f}" for v in PUBLISHED_RING_PCT["TargetDiff"]))
    print("  ref ring % ours " + " ".join(f"{ref['ring_size_pct'][s]:5.1f}" for s in sizes))
    print("        published " + " ".join(f"{v:5.1f}" for v in PUBLISHED_RING_PCT["Reference ligand"]))
    ok = d_jsd <= SELFCHECK_TOL_JSD and d_ring <= SELFCHECK_TOL_RING
    print(f"  max |Δ| bond JSD {d_jsd:.4f} (tol {SELFCHECK_TOL_JSD}), "
          f"ring % {d_ring:.2f} (tol {SELFCHECK_TOL_RING}) -> {'PASS' if ok else 'FAIL'}")
    return {"n_pockets": 100, "n_mols": td["n_mols"], "bond_jsd": dict(zip(BOND_TYPES, got)),
            "ring_size_pct": td["ring_size_pct"], "reference_ring_size_pct": ref["ring_size_pct"],
            "max_abs_diff_bond_jsd": d_jsd, "max_abs_diff_ring_pct": d_ring, "pass": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=ROOT",
                    help="an arm; LABEL should be its figures/draw.py ARMS label. Repeatable.")
    ap.add_argument("--targets", default=str(P79_JSON),
                    help="JSON list of target_* dirs every arm is scored over "
                         "(default: the 79 electron-density pockets)")
    ap.add_argument("--reference-dir", default=str(REFERENCE_DIR),
                    help="CrossDocked test set: <pocket>/<ligand>.sdf, 100 ligands")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--selfcheck", action="store_true",
                    help="also score TargetDiff over all 100 pockets against the published "
                         "tables; exit 1 if it misses")
    args = ap.parse_args()
    if not args.run and not args.selfcheck:
        ap.error("nothing to do: give --run and/or --selfcheck")

    t0 = time.time()
    targets = json.load(open(args.targets))
    ref_files = sorted(glob.glob(os.path.join(args.reference_dir, "*", "*.sdf")))
    if len(ref_files) != 100:
        raise SystemExit(f"expected the 100 CrossDocked test ligands under "
                         f"{args.reference_dir}, found {len(ref_files)}")

    jobs = {"__reference__": ref_files}
    arm_roots = {}
    for spec in args.run:
        label, root = spec.split("=", 1)
        if label in arm_roots:
            ap.error(f"--run {label}: given twice")
        arm_roots[label] = os.path.abspath(root)
        jobs[label] = target_sdfs(arm_roots[label], targets)
    print(f"profiling {len(jobs) - 1} arms over {len(targets)} pockets + "
          f"{len(ref_files)} reference ligands ({sum(map(len, jobs.values()))} SDFs, "
          f"{args.workers} workers)", flush=True)
    profiles = run_profiles(jobs, args.workers)

    ref_profile = profiles.pop("__reference__")
    arms = {}
    for label, prof in profiles.items():
        arms[label] = {"root": arm_roots[label], "n_pockets": len(targets),
                       **summarise(prof, ref_profile)}
    reference = {"label": "Reference ligand", "source": os.path.abspath(args.reference_dir),
                 "n_files": len(ref_files), **summarise(ref_profile)}
    if arms:
        print_tables(reference, arms)

    result = {
        "protocol": {
            "reference": "CrossDocked2020 test set, 100 ligands (TargetDiff/VoxBind protocol)",
            "pockets": os.path.abspath(args.targets),
            "molecules": "RDKit-sanitisable, single connected component",
            "jsd": "scipy.spatial.distance.jensenshannon (JS distance, natural log), "
                   "as printed by TargetDiff and VoxBind",
            "bond_bins": BOND_BINS.tolist(),
            "bond_types": {k: list(v) for k, v in BOND_TYPES.items()},
            "pair_bins": {k: v.tolist() for k, v in PAIR_BINS.items()},
            "atom_types": list(ATOM_TYPES.values()),
            "arom_frac_edges": AROM_EDGES.tolist(),
            "ring_size_pct": "percent of all rings (RDKit AtomRings), sizes 3-9 and 10+",
        },
        "published": {"bond_jsd_columns": list(BOND_TYPES), "bond_jsd": PUBLISHED_BOND_JSD,
                      "ring_size_pct_columns": [str(s) for s in RING_TABLE_SIZES],
                      "ring_size_pct": PUBLISHED_RING_PCT},
        "reference": reference,
        "arms": arms,
    }
    ok = True
    if args.selfcheck:
        result["selfcheck"] = selfcheck(ref_profile, args.workers)
        ok = result["selfcheck"]["pass"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(f"\nJSON -> {args.out}  ({time.time() - t0:.0f}s)")
    if not ok:
        sys.exit(1)
    print("CROSSDOCKED_JSD_DONE")


if __name__ == "__main__":
    main()
