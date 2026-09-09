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
    /opt/conda/envs/voxbind/bin/python \
        notebook/html/260910/fig-ref-ligand-similarity/build_reference_similarity.py
    ... --full           also emit the appendix fingerprints (MACCS/AtomPair/RDKit/Dice)
    ... --with-3d        also emit 3D shape Tanimoto (slower; needs conformers)
    ... --own-pockets    score each method on all of its own pockets instead of on the
                         set shared by every method (cross-method comparison then is
                         NOT apples to apples -- the n pockets column tells you)
    ... --from-metrics   read cached metrics.json where present instead of samples.sdf
    ... --methods "Ours v1" TargetDiff          restrict to a subset (plain labels)

ON ANOTHER SERVER (for the baselines that are not on this box)
    Copy this file over, point METHODS at that machine's sample roots -- any directory
    holding target_*/ subdirs with a samples.sdf and the pocket's reference ligand
    beside it (a *_ref.sdf, or the *_lig_*.sdf next to *_pocket10.pdb) -- and run it
    with an rdkit-bearing python. A method may be given a LIST of roots when its run was
    sharded across GPUs. Then copy the resulting reference_similarity.json back here and
    merge, or just paste the printed rows.

    python build_reference_similarity.py --methods AR Pocket2Mol DiffSBDD DecompDiff

OUTPUTS (all beside this script, in notebook/html/260910/fig-ref-ligand-similarity/)
    reference_similarity.json        per-pocket values + macro-averaged summary
    reference_similarity.csv         one row per method (mean / median / max)
    reference_similarity_table.html  HTML fragment, read by results2latex.ipynb
    reference_similarity.tex         the paper table: methods as rows, metrics as columns
    reference_similarity_wrap.tex    same rows as a \resizebox'd wraptable (needs wrapfig)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
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
REPO = "/home1/irteam/VoxBind"    # the training split is addressed from the repo root,
                                  # not the cwd, so the script runs from any directory
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
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=0.9}": f"{E}/_vanilla_ep923/samples/full_eval_ep923",
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=1.0}": f"{E}/exp_sig1.0_350ep/samples/full_eval_ep349",
    # Same run the de novo Vina table calls "Ours · v1"; the label matches so the two
    # paper tables name the same model the same way.
    "Ours\\textsubscript{\\scriptsize v1}": f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
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
    "Ours\\textsubscript{\\scriptsize v1}": "Ours v1",
}
# AR / Pocket2Mol / DiffSBDD / DecompDiff arrive in a different shape. Their molecules are
# not target_*/samples.sdf trees but TargetDiff-style meta bundles -- one list per test
# pocket of {mol, smiles, ligand_filename, pred_pos} -- staged in the git-ignored results/
# bundle by results/dropbox_pull_baselines.sh. A bundle comes in up to three parts: the
# first 50 molecules per pocket, _part2 with the rest, and _gap with any re-run fill.
#
# Their molecule counts over the 79 shared pockets reproduce the aggregates the Blackwell
# box reported to the digit (7655 / 7772 / 7720 / 6427), which is what says these are the
# same molecules that run measured rather than a different sample of them.
RESULTS = f"{REPO}/results/task2-drugdesign"
META_METHODS = {
    "AR":         f"{RESULTS}/AR/samples/meta/AR",
    "Pocket2Mol": f"{RESULTS}/Pocket2Mol/samples/meta/Pocket2Mol",
    "DiffSBDD":   f"{RESULTS}/DiffSBDD/samples/meta/DiffSBDD",
    "DecompDiff": f"{RESULTS}/DecompDiff/samples/meta/DecompDiff_ref_prior",
}
META_PARTS = ("", "_part2", "_gap")

# A meta bundle carries no reference ligand, so it borrows one per pocket from a run whose
# target dirs cover the whole test set. The vanilla run has all 100; the table's 79 are a
# subset of those.
REF_ROOT = f"{E}/_vanilla_ep923/samples/full_eval_ep923"

HTML_LABEL = {
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=0.9}": 'VoxBind<sub class="sc">σ=0.9</sub>',
    "VoxBind\\textsubscript{\\scriptsize $\\sigma$=1.0}": 'VoxBind<sub class="sc">σ=1.0</sub>',
    "Ours\\textsubscript{\\scriptsize v1}": 'Ours<sub class="sc">v1</sub>',
}

# MEASURED ON THE OTHER BOX. AR / Pocket2Mol / DiffSBDD / DecompDiff were sampled on the
# Blackwell (sm_120) machine and their molecules are not here, so these are the numbers
# that run reported back (2026-09-05) rather than anything this file computes. The request
# that produced them is blackwell_similarity_request.md, beside this script.
#
# WHAT TIES THEM TO OUR POCKET SET: only aggregates came back, so the intersection cannot
# be re-derived here -- but the molecule counts can be checked, and they match exactly.
# The whole-receptor PoseCheck run over our own 79 density pockets counts 7,655 / 7,772 /
# 7,720 / 6,427 molecules for these same four methods, which is what these rows say. Same
# pockets, same samples. If a pocket subset ever changes, ask for the per_pocket block of
# that machine's reference_similarity.json instead of these aggregates.
REMOTE_SOURCE = "Blackwell sm_120 box, aggregates only, reported 2026-09-05"
REMOTE = {
    "AR":         {"n_pockets": 79, "n_mols": 7655, "ecfp4_mean": 0.100,
                   "ecfp4_median": 0.096, "ecfp4_max": 0.251, "scaffold_match": 0.0104},
    "Pocket2Mol": {"n_pockets": 79, "n_mols": 7772, "ecfp4_mean": 0.097,
                   "ecfp4_median": 0.092, "ecfp4_max": 0.239, "scaffold_match": 0.0112},
    "DiffSBDD":   {"n_pockets": 79, "n_mols": 7720, "ecfp4_mean": 0.089,
                   "ecfp4_median": 0.085, "ecfp4_max": 0.227, "scaffold_match": 0.0055},
    # That run used DecompDiff's reference-prior variant. The row is labelled plainly, as
    # the de novo Vina table labels it; the variant is recorded in the json.
    "DecompDiff": {"n_pockets": 79, "n_mols": 6427, "ecfp4_mean": 0.152,
                   "ecfp4_median": 0.137, "ecfp4_max": 0.385, "scaffold_match": 0.0217},
}
REMOTE_NOTES = {"DecompDiff": "reference-prior variant"}

# The paper table's row order, top to bottom, by plain label. Everything above the rule is
# a baseline -- VoxBind included, since it is our own model without the density input.
ROW_ORDER = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "TargetDiff", "FuncBind",
             "VoxBind σ=0.9", "VoxBind σ=1.0", "Ours v1"]
# Row labels that go below the rule, matched by prefix.
OURS_PREFIXES = ("Ours",)


def table_rows(summary):
    """summary + the rows measured elsewhere, in ROW_ORDER.

    Only the table writers merge the two. reference_similarity.{json,csv} keep the remote
    rows in their own block, so a number this script measured is never silently stacked
    with one that was reported to us.
    """
    # measured here wins: REMOTE only fills in a method this run could not compute.
    merged = {**{k: dict(v) for k, v in REMOTE.items()}, **summary}
    by_plain = {PLAIN.get(k, k): k for k in merged}
    out = {by_plain[name]: merged[by_plain[name]] for name in ROW_ORDER if name in by_plain}
    for key, vals in merged.items():        # anything ROW_ORDER does not name, kept at the end
        out.setdefault(key, vals)
    return out


def html_label(key: str) -> str:
    return HTML_LABEL.get(key, PLAIN.get(key, key))


# NOVELTY. The reference-ligand columns answer "is this a copy of the crystal ligand",
# which nobody claimed it was; these answer the question a reviewer actually asks, "did
# the model memorise its training set". Definitions follow MOSES (Polykovskiy et al.,
# Front. Pharmacol. 2020) and GuacaMol (Brown et al., JCIM 2019): Novelty is the fraction
# of generated molecules whose canonical SMILES is absent from the training set, and SNN
# the mean over generated molecules of the Tanimoto to their NEAREST training molecule.
# Scaffold novelty is the same fraction at Bemis-Murcko level.
#
# The training set is the 100,000 CrossDocked pairs in split_by_name.pt -- the same split
# the model trained on, so a hit is a genuine memorisation. Parsing them costs ~40 s and
# the fingerprints ~25 MB, so the index is cached; delete the cache after changing the
# split or the fingerprint.
TRAIN_SPLIT = f"{REPO}/voxbind/dataset/data/split_by_name.pt"
TRAIN_ROOT = f"{REPO}/voxbind/dataset/data/crossdocked_pocket10"
TRAIN_CACHE = f"{REPO}/voxbind/dataset/data/.novelty_train_index_ecfp4.pkl"

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
# What the table calls them. The field names stay mean/median so the json and csv keep
# saying what they hold; only the printed header is abbreviated.
STAT_LABEL = {"mean": "Avg.", "median": "Med."}
STAT_FN = {"mean": st.mean, "median": st.median, "max": max}


def train_index(split=TRAIN_SPLIT, root=TRAIN_ROOT, cache=TRAIN_CACHE):
    """(canonical SMILES set, BM scaffold set, [ECFP4 fps]) for the training ligands."""
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            smi, scaf, fps = pickle.load(fh)
        print(f"  train index (cached): {len(smi):,} molecules, {len(scaf):,} scaffolds")
        return smi, scaf, fps
    import torch
    pairs = torch.load(split, weights_only=False)["train"]
    smi, scaf, fps, seen = set(), set(), [], 0
    for _, lig in pairs:
        path = os.path.join(root, lig)
        if not os.path.exists(path):
            continue
        for mol in Chem.SDMolSupplier(path, sanitize=True):
            if mol is None:
                break
            can = Chem.MolToSmiles(mol)
            seen += 1
            if can not in smi:          # CrossDocked repeats a ligand across poses
                smi.add(can)
                fps.append(_MORGAN.GetFingerprint(mol))
                sc = bm_scaffold(mol)
                if sc is not None:
                    scaf.add(sc)
            break                       # one molecule per docked-pose file
    print(f"  train index: {seen:,} files -> {len(smi):,} unique molecules, "
          f"{len(scaf):,} scaffolds")
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump((smi, scaf, fps), fh, protocol=4)
    return smi, scaf, fps


_TRAIN = None        # set before the pool forks, so workers inherit it without pickling


def _novelty_pocket(task):
    """(novelty, snn, scaffold novelty) for one pocket's molecules."""
    smiles, scaffolds, fps = task
    train_smi, train_scaf, train_fps = _TRAIN
    novel = sum(1 for c in smiles if c not in train_smi) / len(smiles)
    # Molecules RDKit could not scaffold are dropped from the denominator rather than
    # counted as novel -- an unscaffoldable molecule is a parse result, not a finding.
    scaffolded = [s for s in scaffolds if s is not None]
    scaf_novel = (sum(1 for s in scaffolded if s not in train_scaf) / len(scaffolded)
                  if scaffolded else None)
    snn = st.mean([max(DataStructs.BulkTanimotoSimilarity(f, train_fps)) for f in fps])
    return novel, snn, scaf_novel


def add_novelty(per_method, workers):
    """Fill novelty / snn / scaffold_novelty on every pocket, then drop the raw carriers.

    Done as a second pass rather than inside method_values so the 8 ms-per-molecule
    nearest-neighbour search over 90k training fingerprints can be spread across cores;
    serially it is ~5 minutes for the five local methods.
    """
    tasks, where = [], []
    for label, per_pocket in per_method.items():
        for key, rec in per_pocket.items():
            if rec.get("_smiles"):
                tasks.append((rec["_smiles"], rec["_scaffolds"], rec["_fps"]))
                where.append((label, key))
    if not tasks:
        return
    print(f"  novelty: {len(tasks)} pockets, "
          f"{sum(len(t[0]) for t in tasks):,} molecules, {workers} workers")
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_novelty_pocket, tasks, chunksize=1)
    for (label, key), (novel, snn, scaf_novel) in zip(where, results):
        rec = per_method[label][key]
        rec["novelty"], rec["snn"], rec["scaffold_novelty"] = novel, snn, scaf_novel
    for per_pocket in per_method.values():          # keep them out of the json
        for rec in per_pocket.values():
            for k in ("_smiles", "_scaffolds", "_fps"):
                rec.pop(k, None)


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
def pocket_record(ref_mol, mols, fp_keys, want_3d: bool, want_novelty: bool):
    """One pocket's row. Shared by the samples.sdf path and the meta-bundle path, so a
    baseline is scored by exactly the same code as our own runs."""
    rec = {"n_mols": len(mols)}

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

    if want_novelty:
        # Carried, not computed: the nearest-neighbour search runs in a pool later.
        rec["_smiles"] = [Chem.MolToSmiles(m) for m in mols]
        rec["_scaffolds"] = [bm_scaffold(m) for m in mols]
        rec["_fps"] = [_MORGAN.GetFingerprint(m) for m in mols]

    ref_scaf = bm_scaffold(ref_mol)
    if ref_scaf is not None:
        rec["scaffold_match"] = sum(1 for m in mols if bm_scaffold(m) == ref_scaf) / len(mols)

    if want_3d:
        shapes = [s for s in (shape_tanimoto(m, ref_mol) for m in mols) if s is not None]
        if shapes:
            for stat in ALL_STATS:
                rec[f"shape3d_{stat}"] = STAT_FN[stat](shapes)
    return rec


def method_values(roots, fp_keys, want_3d: bool, from_metrics: bool, want_novelty=False):
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
            rec = pocket_record(ref_mol, mols, fp_keys, want_3d, want_novelty)
            rec["target"], rec["root"] = t, root

            if key in out:
                print(f"      warning: pocket {key} seen twice ({out[key]['root']} and {root});"
                      " keeping the first")
                continue
            out[key] = rec
    return out


def reference_index(root=REF_ROOT):
    """{pocket_key: crystal ligand} from a run whose target dirs cover the test set.

    The meta bundles carry only generated molecules, so the reference every
    similarity is measured against comes from here -- the same *_lig_*.sdf file
    load_from_sdf() picks for our own runs, so the two paths compare against the
    identical molecule and never quietly diverge.
    """
    refs = {}
    for t in sorted(d for d in os.listdir(root) if d.startswith("target_")):
        tdir = os.path.join(root, t)
        if not os.path.isdir(tdir):
            continue
        key = pocket_key(tdir)
        loaded = load_from_sdf(tdir) if key else None
        if loaded is not None:
            refs[key] = loaded[0]
    return refs


def load_meta(stem):
    """{pocket_key: [mols]} merged over a meta bundle's parts.

    A molecule that will not sanitise is dropped, the same silent drop
    Chem.SDMolSupplier(sanitize=True) makes on the samples.sdf path.
    """
    import torch
    per = {}
    for suffix in META_PARTS:
        path = f"{stem}{suffix}.pt"
        if not os.path.exists(path):
            continue
        for pocket in torch.load(path, weights_only=False):
            for entry in pocket:
                mol = entry.get("mol")
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue
                key = os.path.basename(entry["ligand_filename"])[: -len(".sdf")] + "_pocket10"
                per.setdefault(key, []).append(mol)
    return per


def method_values_meta(stem, fp_keys, want_3d: bool, want_novelty, refs):
    """{pocket_key: {...}} for a method that ships as a meta bundle."""
    out = {}
    for key, mols in load_meta(stem).items():
        ref_mol = refs.get(key)
        if ref_mol is None or not mols:
            continue
        rec = pocket_record(ref_mol, mols, fp_keys, want_3d, want_novelty)
        rec["target"], rec["root"] = key, os.path.dirname(stem)
        out[key] = rec
    return out


def macro(per_pocket, field):
    vals = [r[field] for r in per_pocket.values() if r.get(field) is not None]
    return st.mean(vals) if vals else None


# ── emitters ──────────────────────────────────────────────────────────────────
def metric_groups(fp_keys, want_3d, want_novelty=False):
    """[(group name, [(field, stat label, formatter), ...])] in table order.

    A group with more than one stat becomes a \\multirow block in LaTeX and a colspan
    header in HTML; a group with a single unnamed stat (Scaffold match) spans the two
    stub columns instead.
    """
    groups = [(FPS[fk][0], [(f"{fk}_{s}", STAT_LABEL.get(s, s), "f3") for s in TABLE_STATS])
              for fk in fp_keys]
    groups.append(("Scaffold match", [("scaffold_match", "", "pct")]))
    if want_novelty:
        # Two groups, because they are two different quantities: a rate of molecules that
        # are new, and how close the nearest training molecule is when they are not.
        groups.append(("Novelty vs. train", [("novelty", "SMILES", "pct"),
                                             ("scaffold_novelty", "Scaffold", "pct")]))
        groups.append(("SNN", [("snn", "", "f3")]))
    if want_3d:
        groups.append(("3D shape", [(f"shape3d_{s}", s, "f3") for s in TABLE_STATS]))
    return groups


def columns(fp_keys, want_3d, want_novelty=False):
    """Flattened (field, header, formatter) view of metric_groups, for csv/console."""
    return [(field, f"{name} {stat}".strip(), kind)
            for name, stats in metric_groups(fp_keys, want_3d, want_novelty)
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
    """Methods as rows, metrics as columns, with a two-row grouped header — the shape
    results.html uses and the shape results2latex.ipynb renders straight through."""
    top, sub = [], []
    for name, stats in groups:
        if len(stats) == 1 and not stats[0][1]:
            top.append(f'<th rowspan="2">{name}</th>')
        else:
            top.append(f'<th class="grp" colspan="{len(stats)}">{name}</th>')
            sub.extend(f"<th>{stat}</th>" for _, stat, _ in stats)
    rows = []
    for label, vals in summary.items():
        if vals.get("pending"):
            cells = "".join('<td><span class="tbd">TBA</span></td>'
                            for _, stats in groups for _ in stats)
            rows.append(f'            <tr class="pending">'
                        f'<td class="col-method">{html_label(label)}</td>{cells}</tr>')
            continue
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


CAPTION = r"\textbf{Reference-ligand similarity}, metrics averaged over pockets."
LABEL = "tab:result-drug-reference-similarity"


def latex_table(summary, groups, wrap: bool = False) -> str:
    """Methods as rows, metrics as columns — the same orientation as the fragment and
    as the de novo Vina table, so the two paper tables read the same way down the page.

    A group with several statistics (ECFP4 -> mean / median) takes a \\multicolumn with a
    \\cmidrule under it; a single-statistic group (Scaffold match) takes a \\multirow that
    spans both header rows, its name broken over two lines with \\shortstack. That is the
    same treatment the Vina table gives its High aff. column. \\shortstack is plain LaTeX,
    so this needs only booktabs and multirow.

    Placeholder rows (PENDING) come out commented, with a trailing note, exactly as the
    de novo table does for arms that are still sampling.

    wrap=True emits the 260820 layout: a \\resizebox'd wraptable for sitting beside the
    body text (needs wrapfig), same rows either way.

    results2latex.ipynb renders the same fragment independently; the two outputs are
    meant to agree, so a divergence in either is a one-line diff to catch.
    """
    labels = list(summary)
    n_values = sum(len(stats) for _, stats in groups)
    pad = "        " if wrap else "    "        # \begin{tabular} indent
    row = pad + "    "                          # rows inside it

    top_cells, sub_cells, rules, col = [], [], [], 2
    for name, stats in groups:
        if len(stats) == 1 and not stats[0][1]:
            stacked = r"\\ ".join(rf"\textbf{{{part}}}" for part in name.split(" ", 1))
            top_cells.append(rf"\multirow{{2}}{{*}}{{\shortstack{{{stacked}}}}}")
            sub_cells.append("")
            col += 1
        else:
            top_cells.append(rf"\multicolumn{{{len(stats)}}}{{c}}{{\textbf{{{name}}}}}")
            sub_cells.extend(stat for _, stat, _ in stats)
            rules.append(rf"\cmidrule(lr){{{col}-{col + len(stats) - 1}}}")
            col += len(stats)

    body = [r"\begin{wraptable}{r}{0.35\textwidth}"] if wrap else [r"\begin{table}[t]"]
    if wrap:
        body.append(r"    \vspace{-.15in}")
    body += [
        r"    \centering",
        r"    \caption{",
        "        " + CAPTION,
        r"    }",
        rf"    \label{{{LABEL}}}",
    ]
    if wrap:
        body.append(r"    \resizebox{.98\linewidth}{!}{%")
    body += [
        pad + rf"\begin{{tabular}}{{l{'c' * n_values}}}",
        row + r"\toprule",
        # The stub is empty on the first header row and carries "Method" on the second,
        # so the metric group names sit alone on the top line above their own rule.
        row + "& " + " & ".join(top_cells) + r" \\",
        row + "".join(rules),
        row + " & ".join([r"\textbf{Method}", *sub_cells]) + r" \\",
        row + r"\midrule",
    ]

    is_ours = [m.startswith(OURS_PREFIXES) for m in labels]
    first = is_ours.index(True) if any(is_ours) else None
    split = first if first and all(is_ours[first:]) else None

    n_pending = 0
    for index, label in enumerate(labels):
        if index == split:
            body.append(row + r"\midrule")
        vals = summary[label]
        if vals.get("pending"):
            n_pending += 1
            values = ["TBA"] * n_values
            prefix = "% "
        else:
            values = [fmt_tex(vals.get(field), kind)
                      for _, stats in groups for field, _, kind in stats]
            prefix = ""
        body.append(row + prefix + " & ".join([label, *values]) + r" \\")

    body += [row + r"\bottomrule", pad + r"\end{tabular}"]
    if wrap:
        body += [r"    }", r"    \vspace{-.4in}"]
    if n_pending:
        body.append(f"    % {n_pending} row(s) commented out: no numbers for them yet "
                    "(see REMOTE in build_reference_similarity.py).")
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
    ap.add_argument("--novelty", action="store_true",
                    help="also report novelty and SNN against the CrossDocked training "
                         "split (MOSES/GuacaMol definitions); adds ~1 min on first run")
    ap.add_argument("--novelty-workers", type=int, default=16,
                    help="processes for the nearest-neighbour search (default 16)")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    if args.from_metrics:
        print("  warning: --from-metrics reads a cache that disagrees with the SDFs for "
              "some methods (TargetDiff: 7,287 molecules vs 7,798); rows may sit on "
              "different molecule sets")
    if args.with_3d and args.from_metrics:
        sys.exit("--with-3d needs conformers, which metrics.json does not carry")

    fp_keys = HEADLINE_FPS + (APPENDIX_FPS if args.full else [])
    wanted, wanted_meta = METHODS, META_METHODS
    if args.methods:
        sel = set(args.methods)
        wanted = {k: v for k, v in METHODS.items() if k in sel or PLAIN.get(k) in sel}
        wanted_meta = {k: v for k, v in META_METHODS.items() if k in sel}
        if not wanted and not wanted_meta:
            sys.exit(f"no methods matched {args.methods}; known: "
                     + ", ".join([PLAIN.get(k, k) for k in METHODS] + list(META_METHODS)))

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
        per_method[label] = method_values(live, fp_keys, args.with_3d, args.from_metrics,
                                          args.novelty)
        print(f"      {len(per_method[label])} pockets, "
              f"{sum(r['n_mols'] for r in per_method[label].values())} molecules")

    if wanted_meta:
        # These four ship as meta bundles and borrow their reference ligand per pocket, so
        # the index is built once and shared. A bundle that has not been pulled is skipped
        # with a pointer at the script that pulls it -- results/ is git-ignored.
        refs = None
        for label, stem in wanted_meta.items():
            parts = [f"{stem}{suf}.pt" for suf in META_PARTS]
            live = [f for f in parts if os.path.exists(f)]
            if not live:
                print(f"  skip {label}: no meta bundle at {stem}*.pt "
                      "(pull it with results/dropbox_pull_baselines.sh)")
                continue
            if refs is None:
                refs = reference_index()
                print(f"  reference ligands <- {REF_ROOT} ({len(refs)} pockets)")
            print(f"  {label} <- " + ", ".join(os.path.basename(f) for f in live))
            per_method[label] = method_values_meta(stem, fp_keys, args.with_3d,
                                                   args.novelty, refs)
            print(f"      {len(per_method[label])} pockets, "
                  f"{sum(r['n_mols'] for r in per_method[label].values())} molecules")

    if not per_method:
        sys.exit("no methods produced any pockets")

    if args.novelty:
        global _TRAIN
        _TRAIN = train_index()
        add_novelty(per_method, args.novelty_workers)

    shared = set.intersection(*(set(v) for v in per_method.values()))
    if not args.own_pockets and not shared:
        sys.exit("the methods share no pockets; re-run with --own-pockets")
    print(f"  {len(shared)} pockets shared by all {len(per_method)} methods"
          + (" (ignored: --own-pockets)" if args.own_pockets else ""))

    fields = [f"{fk}_{stat}" for fk in fp_keys for stat in ALL_STATS] + ["scaffold_match"]
    if args.novelty:
        fields += ["novelty", "scaffold_novelty", "snn"]
    if args.with_3d:
        fields += [f"shape3d_{stat}" for stat in ALL_STATS]

    summary = {}
    for label, per_pocket in per_method.items():
        sub = per_pocket if args.own_pockets else {k: v for k, v in per_pocket.items() if k in shared}
        summary[label] = {f: macro(sub, f) for f in fields}
        summary[label]["n_pockets"] = len(sub)
        summary[label]["n_mols"] = sum(r["n_mols"] for r in sub.values())

    groups = metric_groups(fp_keys, args.with_3d, args.novelty)
    cols = columns(fp_keys, args.with_3d, args.novelty)
    rows = table_rows(summary)
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
            # Kept in its own block: summary and per_pocket are what THIS run measured.
            "remote": {"source": REMOTE_SOURCE, "notes": REMOTE_NOTES, "values": REMOTE},
        }, fh, indent=2)

    with open(os.path.join(out, "reference_similarity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_pockets", "n_mols"] + [f for f, _, _ in cols] + ["source"])
        for label, vals in rows.items():
            plain = PLAIN.get(label, label)
            w.writerow([plain, vals["n_pockets"], vals["n_mols"]]
                       + [vals.get(f) for f, _, _ in cols]
                       + ["this machine" if label in summary else "blackwell"])

    # The table writers also carry the rows measured on the other box; above, the json
    # and csv keep those separate from what this run computed.
    rows = table_rows(summary)
    write_html(os.path.join(out, "reference_similarity_table.html"),
               rows, groups, args.own_pockets, len(shared))
    write_tex(os.path.join(out, "reference_similarity.tex"), rows, groups)
    write_tex(os.path.join(out, "reference_similarity_wrap.tex"), rows, groups, wrap=True)

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
