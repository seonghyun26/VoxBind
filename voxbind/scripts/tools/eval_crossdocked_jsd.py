#!/usr/bin/env python3
"""eval_crossdocked_jsd.py — how closely generated molecules match CrossDocked ligands'
geometry and composition: the TargetDiff / VoxBind distribution metrics.

    python voxbind/scripts/tools/eval_crossdocked_jsd.py \
        --run AR=<root> --run CoDE=<root> ... --out crossdocked_jsd.json [--selfcheck]

Each <root> holds target_*/samples.sdf. `voxbind/scripts/89_eval_crossdocked_jsd.sh` runs
it over the eight arms figures/draw.py draws; `draw.py jsd` then plots the JSON and
figures/fig-jsd/jsd-tables.ipynb tabulates it.

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
                molecule's heavy atoms that are aromatic (VoxBind Fig. 10). Alongside it,
                the fraction of a molecule's RINGS that are aromatic (every ring bond
                aromatic), over molecules with at least one ring, and exact means of both.

THE REFERENCES. Three are built; the first is the one the figures and main tables use.

  reference            ALL CrossDocked2020 ligands the models were trained and tested on:
                       split_by_name.pt train + test, 100,100 ligand files. Bond lengths
                       and composition are chemistry, not properties of 100 test pockets,
                       and 100 ligands are too few to estimate them: a model that sampled
                       the training distribution exactly would still score C=N JSD 0.52
                       and C=C 0.46 against the test set (16 and 40 bonds), which is most of
                       every method's score in those columns. DUPLICATES: the 100,000
                       training files hold 8,765 distinct molecules -- one ligand is
                       cross-docked into up to 869 pockets -- so each distinct molecule
                       (canonical isomeric SMILES) weighs 1, averaged over its poses. No
                       pose is picked, and no ligand counts 869 times.
  reference_pose_weighted
                       the same files, every pose weighing 1: the distribution training
                       actually saw. Kept, with every arm's JSD against it (`jsd_pose_
                       weighted`), as the sensitivity check on the de-duplication.
  reference_test       the 100 test ligands: the PUBLISHED protocol (arms' `jsd_test`).
                       Scoring TargetDiff's own 9,878 molecules against them reproduces its
                       published row (C-C .370 / C=N .548 / C:N .235 for .369 / .550 /
                       .235); the histograms TargetDiff ships in `eval_bond_length_config`
                       do not (.299 / .165 / .131) -- they are the training set. `--selfcheck`
                       re-runs that comparison, and also scores the shipped histograms
                       against `reference_pose_weighted`, which should nearly coincide.

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
TEST_DIR = REPO / "targetdiff" / "data" / "test_set" / "test_set"
SPLIT = REPO / "voxbind" / "dataset" / "data" / "split_by_name.pt"
CROSSDOCKED_ROOT = REPO / "voxbind" / "dataset" / "data" / "crossdocked_pocket10"
TARGETDIFF_CONFIG = REPO / "targetdiff" / "utils" / "evaluation" / "eval_bond_length_config.py"
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
AROM_RING_BINS = 10          # aromatic share of a molecule's rings, [k/10, (k+1)/10), last closed

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

SCALAR_KEYS = ("n_entries", "n_unparseable", "n_disconnected", "n_mols",
               "n_with_heavy", "n_with_rings", "arom_atom_sum", "arom_ring_sum")


def zero_profile(dtype=np.int64):
    """An empty profile. int for counting molecules; float for a weighted reference."""
    return {
        "n_entries": 0, "n_unparseable": 0, "n_disconnected": 0, "n_mols": 0,
        "bond": {k: np.zeros(len(BOND_BINS) + 1, dtype) for k in BOND_TYPES},
        "pair": {k: np.zeros(len(b) + 1, dtype) for k, b in PAIR_BINS.items()},
        "atoms": collections.Counter(), "ring_sizes": collections.Counter(),
        "n_rings": collections.Counter(), "arom": np.zeros(len(AROM_EDGES) - 1, dtype),
        "arom_ring": np.zeros(AROM_RING_BINS, dtype),
        "n_with_heavy": 0, "n_with_rings": 0, "arom_atom_sum": 0.0, "arom_ring_sum": 0.0,
    }


def accumulate(acc, p, w=1):
    """acc += w * p, field by field. Every field is additive, which is what lets a run be
    the sum of its files and a weighted reference the weighted sum of its ligands."""
    for k in SCALAR_KEYS:
        acc[k] += w * p[k]
    for k in ("bond", "pair"):
        for name in acc[k]:
            acc[k][name] += w * p[k][name]
    for k in ("atoms", "ring_sizes", "n_rings"):
        for key, v in p[k].items():
            acc[k][key] += w * v
    acc["arom"] += w * p["arom"]
    acc["arom_ring"] += w * p["arom_ring"]
    return acc


def merge(profiles):
    acc = zero_profile()
    for p in profiles:
        accumulate(acc, p)
    return acc


def profile_sdf(path):
    """Additive counts for one SDF: every histogram is a count vector, so a run's profile
    is the sum over its files and no per-molecule list ever crosses a process boundary."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    p = zero_profile()
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
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        p["n_rings"][len(rings)] += 1
        p["ring_sizes"].update(len(r) for r in rings)
        if heavy:
            # Integer arithmetic, not np.histogram over AROM_EDGES: linspace's 0.3 edge is
            # 0.30000000000000004, so a molecule exactly 6/20 aromatic fell one bin low --
            # 3.2% of molecules. The last bin is closed, so a fully aromatic molecule counts.
            n_arom = sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in heavy)
            nbin = len(AROM_EDGES) - 1
            p["arom"][min(n_arom * nbin // len(heavy), nbin - 1)] += 1
            p["n_with_heavy"] += 1
            p["arom_atom_sum"] += n_arom / len(heavy)
        if rings:
            # The share of a molecule's RINGS that are aromatic (every ring bond aromatic).
            # VoxBind Fig. 10 plots aromatic ATOMS; this is the ring-level companion, and it
            # is undefined for a ring-free molecule, which is left out of it.
            n_arom_rings = sum(all(mol.GetBondWithIdx(b).GetIsAromatic() for b in br)
                               for br in ring_info.BondRings())
            p["n_with_rings"] += 1
            p["arom_ring_sum"] += n_arom_rings / len(rings)
            p["arom_ring"][min(n_arom_rings * AROM_RING_BINS // len(rings),
                               AROM_RING_BINS - 1)] += 1
    return p


def ligand_key(path):
    """The molecule a CrossDocked ligand file holds, as canonical isomeric SMILES of its
    heavy-atom graph -- the identity the de-duplication groups poses by. None for a file
    the profile would not score (unparseable, disconnected, or not exactly one entry)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True))
    if len(mols) != 1 or mols[0] is None or len(Chem.GetMolFrags(mols[0])) > 1:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(mols[0]))


def profile_chunk(items):
    """(de-duplicated, pose-weighted) profiles of a chunk of (ligand file, weight). Summed
    inside the worker, so 100,100 per-file profiles never cross a process boundary."""
    unique, poses = zero_profile(np.float64), zero_profile()
    for path, w in items:
        p = profile_sdf(path)
        accumulate(unique, p, w)
        accumulate(poses, p)
    return unique, poses


def crossdocked_reference(split_path, root, workers, chunk=400):
    """(de-duplicated profile, pose-weighted profile, provenance) for every ligand file of
    the train and test lists in split_by_name.pt."""
    import torch
    split = torch.load(split_path, weights_only=False)
    files = [os.path.join(root, lig) for part in ("train", "test") for _, lig in split[part]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(files)} split ligands missing under {root} "
                         f"(first: {missing[0]})")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        keys = list(ex.map(ligand_key, files, chunksize=256))
    poses_of = collections.Counter(k for k in keys if k is not None)
    items = [(f, 1.0 / poses_of[k]) for f, k in zip(files, keys) if k is not None]
    unique, poses = zero_profile(np.float64), zero_profile()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for u, p in ex.map(profile_chunk, [items[i:i + chunk] for i in range(0, len(items), chunk)]):
            accumulate(unique, u)
            accumulate(poses, p)
    info = {"split": os.path.abspath(split_path), "ligand_root": os.path.abspath(root),
            "n_train": len(split["train"]), "n_test": len(split["test"]), "n_files": len(files),
            "n_files_not_scored": len(files) - len(items), "n_unique_ligands": len(poses_of),
            "max_poses_per_ligand": max(poses_of.values())}
    return unique, poses, info


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


def _num(x):
    """A count for JSON: an int where it is whole (every generated arm), otherwise the
    weighted count of the de-duplicated reference, to 4 places."""
    x = float(x)
    return int(x) if x.is_integer() else round(x, 4)


def summarise(p):
    """A profile as JSON: the counts behind every figure and table."""
    atom_total = sum(p["atoms"].values())
    listed = np.array([p["atoms"].get(z, 0) for z in ATOM_TYPES], float)
    return {
        "n_entries": _num(p["n_entries"]), "n_unparseable": _num(p["n_unparseable"]),
        "n_disconnected": _num(p["n_disconnected"]), "n_mols": _num(p["n_mols"]),
        "bond_counts": {k: [_num(x) for x in v] for k, v in p["bond"].items()},
        "bond_n": {k: _num(v.sum()) for k, v in p["bond"].items()},
        "pair_counts": {k: [_num(x) for x in v] for k, v in p["pair"].items()},
        "pair_n": {k: _num(v.sum()) for k, v in p["pair"].items()},
        "atom_counts": {ATOM_TYPES.get(z, str(z)): _num(n) for z, n in sorted(p["atoms"].items())},
        "atom_frac": {s: (float(listed[i] / listed.sum()) if listed.sum() else 0.0)
                      for i, s in enumerate(ATOM_TYPES.values())},
        "atom_other_frac": (1 - float(listed.sum()) / atom_total) if atom_total else 0.0,
        "ring_size_counts": {str(s): _num(n) for s, n in sorted(p["ring_sizes"].items())},
        "ring_size_pct": ring_share(p["ring_sizes"]),
        "n_rings_counts": {str(s): _num(n) for s, n in sorted(p["n_rings"].items())},
        "arom_frac_counts": [_num(x) for x in p["arom"]],
        "mean_arom_atom_frac": (p["arom_atom_sum"] / p["n_with_heavy"]
                                if p["n_with_heavy"] else None),
        "arom_ring_frac_counts": [_num(x) for x in p["arom_ring"]],
        "n_mols_with_rings": _num(p["n_with_rings"]),
        "mean_arom_ring_frac": (p["arom_ring_sum"] / p["n_with_rings"]
                                if p["n_with_rings"] else None),
        "mean_n_rings": (sum(s * n for s, n in p["n_rings"].items()) / p["n_mols"]
                         if p["n_mols"] else None),
    }


def jsd_block(p, ref):
    """Every JSD of profile `p` against reference profile `ref`."""
    bond = {k: _jsd(ref["bond"][k], p["bond"][k]) for k in BOND_TYPES}
    have = [v for v in bond.values() if v is not None]
    listed = np.array([p["atoms"].get(z, 0) for z in ATOM_TYPES], float)
    ref_listed = np.array([ref["atoms"].get(z, 0) for z in ATOM_TYPES], float)
    return {
        "bond": bond,
        "bond_mean": float(np.mean(have)) if len(have) == len(bond) else None,
        "pair": {k: _jsd(ref["pair"][k], p["pair"][k]) for k in PAIR_BINS},
        "atom_type": _jsd(ref_listed, listed),
    }


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


def _cell(v, width, fmt=".3f"):
    return f"{v:{width}{fmt}}" if v is not None else f"{'—':>{width}s}"


def print_tables(refs, arms):
    names = list(BOND_TYPES)
    for key, title in (("jsd", f"the CrossDocked train+test ligands ({refs['reference']['n_mols']:,.0f} "
                                f"distinct, de-duplicated) — MAIN"),
                       ("jsd_test", f"the {refs['reference_test']['n_mols']} CrossDocked test ligands "
                                    f"(published protocol)")):
        print(f"\n  bond-distance JSD vs {title}")
        print(f"  {'':18s}" + "".join(f"{n:>7s}" for n in names) + f"{'mean':>7s}")
        for lab, a in arms.items():
            j = a[key]
            print(f"  {lab:18s}" + "".join(_cell(v, 7) for v in j["bond"].values())
                  + _cell(j["bond_mean"], 7))

    print(f"\n  {'':18s}{'mols':>7s}{'discon':>7s}{'All<12Å':>9s}{'C-C<2Å':>8s}{'atom':>7s}"
          f"{'rings':>7s}   | mean bond JSD: main / pose-weighted / test")
    for lab, a in arms.items():
        j = a["jsd"]
        print(f"  {lab:18s}{a['n_mols']:7d}{a['n_disconnected']:7d}"
              f"{_cell(j['pair']['All_12A'], 9)}{_cell(j['pair']['CC_2A'], 8)}"
              f"{_cell(j['atom_type'], 7)}{_cell(a['mean_n_rings'], 7, '.2f')}   |"
              f"{_cell(j['bond_mean'], 7)}{_cell(a['jsd_pose_weighted']['bond_mean'], 7)}"
              f"{_cell(a['jsd_test']['bond_mean'], 7)}")

    sizes = [str(s) for s in RING_TABLE_SIZES] + ["10+ of all"]
    print("\n  % of size-3-9 rings by size (TargetDiff Table 2); last column % of all rings")
    print(f"  {'':22s}" + "".join(f"{s:>6s}" for s in sizes))
    rows = [("Reference (train+test)", refs["reference"]), ("Reference (test)", refs["reference_test"])]
    for lab, a in rows + list(arms.items()):
        print(f"  {lab:22s}" + "".join(f"{a['ring_size_pct'][s]:6.1f}" for s in sizes))


def _shipped_histograms():
    """TargetDiff's shipped bond-length histograms, read out of its config file without
    importing its package; None where the checkout does not carry it."""
    if not TARGETDIFF_CONFIG.exists():
        return None
    ns = {}
    exec(TARGETDIFF_CONFIG.read_text(), {"np": np}, ns)
    return {name: np.asarray(ns["EMPIRICAL_DISTRIBUTIONS"][bt], float)
            for name, bt in BOND_TYPES.items()}


def selfcheck(test_profile, pose_profile, workers):
    """TargetDiff's own samples over all 100 pockets against the published row (the test
    reference), plus the provenance check on the shipped histograms. Returns the report;
    the caller decides whether a miss fails the run."""
    root = BASEDRUG / "eval" / "targetdiff"
    files = sorted(glob.glob(str(root / "target_*" / "samples.sdf")))
    if len(files) != 100:
        raise SystemExit(f"selfcheck: expected 100 TargetDiff pockets under {root}, "
                         f"found {len(files)}")
    td_profile = run_profiles({"TargetDiff": files}, workers)["TargetDiff"]
    td, ref = summarise(td_profile), summarise(test_profile)
    got = [jsd_block(td_profile, test_profile)["bond"][k] for k in BOND_TYPES]
    d_jsd = max(abs(a - b) for a, b in zip(got, PUBLISHED_BOND_JSD["TargetDiff"]))
    sizes = [str(s) for s in RING_TABLE_SIZES]
    d_ring = max(max(abs(td["ring_size_pct"][s] - v)
                     for s, v in zip(sizes, PUBLISHED_RING_PCT["TargetDiff"])),
                 max(abs(ref["ring_size_pct"][s] - v)
                     for s, v in zip(sizes, PUBLISHED_RING_PCT["Reference ligand"])))
    print("\n  SELFCHECK — TargetDiff, 100 pockets, against the published tables (test reference)")
    print("  bond JSD   ours " + " ".join(f"{v:.3f}" for v in got))
    print("        published " + " ".join(f"{v:.3f}" for v in PUBLISHED_BOND_JSD["TargetDiff"]))
    print("  ring %     ours " + " ".join(f"{td['ring_size_pct'][s]:5.1f}" for s in sizes))
    print("        published " + " ".join(f"{v:5.1f}" for v in PUBLISHED_RING_PCT["TargetDiff"]))
    print("  ref ring % ours " + " ".join(f"{ref['ring_size_pct'][s]:5.1f}" for s in sizes))
    print("        published " + " ".join(f"{v:5.1f}" for v in PUBLISHED_RING_PCT["Reference ligand"]))
    ok = d_jsd <= SELFCHECK_TOL_JSD and d_ring <= SELFCHECK_TOL_RING
    print(f"  max |Δ| bond JSD {d_jsd:.4f} (tol {SELFCHECK_TOL_JSD}), "
          f"ring % {d_ring:.2f} (tol {SELFCHECK_TOL_RING}) -> {'PASS' if ok else 'FAIL'}")
    report = {"n_pockets": 100, "n_mols": td["n_mols"], "bond_jsd": dict(zip(BOND_TYPES, got)),
              "ring_size_pct": td["ring_size_pct"], "reference_ring_size_pct": ref["ring_size_pct"],
              "max_abs_diff_bond_jsd": d_jsd, "max_abs_diff_ring_pct": d_ring, "pass": ok}

    shipped = _shipped_histograms()
    if shipped is not None:
        # Not pass/fail: the shipped histograms carry no provenance, so this is how close
        # "the training set" (as TargetDiff built it) sits to ours, bond type by bond type.
        prov = {k: _jsd(shipped[k], pose_profile["bond"][k]) for k in BOND_TYPES}
        print("  shipped eval_bond_length_config vs our pose-weighted train+test, bond JSD: "
              + " ".join(f"{k} {v:.3f}" for k, v in prov.items()))
        report["shipped_histograms_vs_pose_weighted_bond_jsd"] = prov
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=ROOT",
                    help="an arm; LABEL should be its figures/draw.py ARMS label. Repeatable.")
    ap.add_argument("--targets", default=str(P79_JSON),
                    help="JSON list of target_* dirs every arm is scored over "
                         "(default: the 79 electron-density pockets)")
    ap.add_argument("--split", default=str(SPLIT),
                    help="split_by_name.pt whose train + test ligands form the main reference")
    ap.add_argument("--crossdocked-root", default=str(CROSSDOCKED_ROOT),
                    help="crossdocked_pocket10/, holding <pocket>/<ligand>.sdf for the split")
    ap.add_argument("--test-dir", default=str(TEST_DIR),
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
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "*", "*.sdf")))
    if len(test_files) != 100:
        raise SystemExit(f"expected the 100 CrossDocked test ligands under "
                         f"{args.test_dir}, found {len(test_files)}")

    print("profiling the CrossDocked train+test reference ...", flush=True)
    uniq_profile, pose_profile, info = crossdocked_reference(args.split, args.crossdocked_root,
                                                             args.workers)
    print(f"  {info['n_files']:,} ligand files, {info['n_unique_ligands']:,} distinct molecules "
          f"(up to {info['max_poses_per_ligand']} poses of one), "
          f"{info['n_files_not_scored']} not scored  ({time.time() - t0:.0f}s)", flush=True)

    jobs = {"__test__": test_files}
    arm_roots = {}
    for spec in args.run:
        label, root = spec.split("=", 1)
        if label in arm_roots:
            ap.error(f"--run {label}: given twice")
        arm_roots[label] = os.path.abspath(root)
        jobs[label] = target_sdfs(arm_roots[label], targets)
    print(f"profiling {len(jobs) - 1} arms over {len(targets)} pockets + {len(test_files)} "
          f"test ligands ({sum(map(len, jobs.values()))} SDFs, {args.workers} workers)", flush=True)
    profiles = run_profiles(jobs, args.workers)
    test_profile = profiles.pop("__test__")

    arms = {}
    for label, prof in profiles.items():
        arms[label] = {"root": arm_roots[label], "n_pockets": len(targets), **summarise(prof),
                       "jsd": jsd_block(prof, uniq_profile),
                       "jsd_pose_weighted": jsd_block(prof, pose_profile),
                       "jsd_test": jsd_block(prof, test_profile)}
    refs = {
        "reference": {
            "label": "Reference ligand (CrossDocked train+test)", "short": "CrossDocked train+test",
            "weighting": "each distinct molecule (canonical isomeric SMILES) weighs 1, "
                         "averaged over its poses; n_mols and every count are weighted",
            **info, **summarise(uniq_profile)},
        "reference_pose_weighted": {
            "label": "Reference ligand (CrossDocked train+test, pose-weighted)",
            "short": "CrossDocked train+test, pose-weighted", "weighting": "every ligand file weighs 1",
            **info, **summarise(pose_profile)},
        "reference_test": {
            "label": "Reference ligand (CrossDocked test)", "short": "CrossDocked test",
            "source": os.path.abspath(args.test_dir), "n_files": len(test_files),
            **summarise(test_profile)},
    }
    if arms:
        print_tables(refs, arms)

    result = {
        "protocol": {
            "reference": "main: CrossDocked2020 train+test ligands of split_by_name.pt, each distinct "
                         "molecule weighing 1 (arms' `jsd`); also pose-weighted (`jsd_pose_weighted`) "
                         "and the 100 test ligands, the published protocol (`jsd_test`)",
            "pockets": os.path.abspath(args.targets),
            "molecules": "RDKit-sanitisable, single connected component",
            "jsd": "scipy.spatial.distance.jensenshannon (JS distance, natural log), "
                   "as printed by TargetDiff and VoxBind",
            "bond_bins": BOND_BINS.tolist(),
            "bond_types": {k: list(v) for k, v in BOND_TYPES.items()},
            "pair_bins": {k: v.tolist() for k, v in PAIR_BINS.items()},
            "atom_types": list(ATOM_TYPES.values()),
            "arom_frac_edges": AROM_EDGES.tolist(),
            "arom_ring_frac": f"per molecule with >=1 ring: aromatic rings (every ring bond "
                              f"aromatic) / rings, {AROM_RING_BINS} bins [k/{AROM_RING_BINS}, "
                              f"(k+1)/{AROM_RING_BINS}), last bin closed",
            "ring_size_pct": "sizes 3-9: percent of the size-3-9 rings (RDKit AtomRings), as "
                             "TargetDiff Table 2; '10+ of all': percent of all rings",
        },
        "published": {"bond_jsd_columns": list(BOND_TYPES), "bond_jsd": PUBLISHED_BOND_JSD,
                      "ring_size_pct_columns": [str(s) for s in RING_TABLE_SIZES],
                      "ring_size_pct": PUBLISHED_RING_PCT},
        **refs,
        "arms": arms,
    }
    ok = True
    if args.selfcheck:
        result["selfcheck"] = selfcheck(test_profile, pose_profile, args.workers)
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
