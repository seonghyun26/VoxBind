#!/usr/bin/env python3
"""relax_poses.py — every pose staged three ways, for interaction fingerprinting.

    <out>/<Label>-asgen/target_NN.sdf         the pose as generated ("gen" is taken; see STAGES)
    <out>/<Label>-relaxfree/target_NN.sdf     ligand-only UFF, the protein unseen
    <out>/<Label>-relaxpocket/target_NN.sdf   UFF with the pocket10 crop held fixed
    <out>/manifest.json                       the layout compute_baseline_interactions.py reads

VoxBind's Fig. 11/13 report a pose "after a small force-field relaxation". Nothing in this
project stores such a pose -- metrics.json, the baselines' pc_*.pt and the posecheck_<M>.json
exports all carry numbers computed FROM one, never the conformer itself -- so it has to be
produced, and this produces it for all eight methods and the crystal ligands.

TWO RELAXATIONS, BECAUSE "RELAXED" IS AMBIGUOUS AND THE TWO ANSWER DIFFERENT QUESTIONS.

  relaxfree    AddHs(addCoords) -> UFFOptimizeMolecule -> RemoveHs. This is PoseCheck's own
               relax, the one `calculate_strain_energy` measures against, and the definition
               the paper's Fig. 13 scale matches. It never looks at the protein, so a ligand
               is free to relax INTO it: an interaction count that rises here may be the
               ligand overlapping the pocket rather than engaging it. Measured 17 ms/molecule.

  relaxpocket  the ligand minimised in the field of the pocket10 crop, every protein atom
               added as a fixed point. Physically the right thing to fingerprint, but it is
               NOT the paper's definition, so it is computed beside the other rather than
               instead of it. Measured 1.93 s/molecule.

THE CROP IS NOT TRIMMED, and that is deliberate rather than lazy. Cutting the protein to
residues near the ligand is 6x-24x faster (160 atoms 0.30 s/mol, 80 atoms 0.08 s/mol against
440 atoms 1.93 s/mol), and on the median molecule it changes nothing -- but the WORST
molecule moves 1.18 A at a 6 A cut and 3.84 A at 4 A, against a generated-to-relaxed signal
of 1.10 A. A trim would therefore inject an error the size of the effect into the tail, which
is the same mistake as scoring clashes on a crop. Full crop, and pay the 1.93 s.

THE RECEPTOR IS THE pocket10 CROP THROUGHOUT, which is this folder's convention and not a
choice made here: every existing interaction series -- the local arms, the four baselines and
the crystal ligands -- is crop-scored so that they can share an axis, and a relaxed column has
to stand beside the generated one. It is the opposite of the clash figures' whole-receptor
convention. Moving the whole interaction family to the whole receptor is a bigger job than
this one, because the generated series would have to be recomputed too.

TWO PHASES, so that 59k molecules never cross a process boundary. The parent writes every
`-gen` file and the manifest (cheap: a torch load and an SDF write), then a pool relaxes one
(method, pocket) at a time by READING that file back. Each worker therefore loads its pocket's
crop once and reuses it for every molecule in the job, and nothing is pickled but paths.

    /opt/conda/envs/voxbind/bin/python relax_poses.py <out_dir> [--workers 28]
    PATH=/opt/conda/envs/moleval/bin:$PATH /opt/conda/envs/moleval/bin/python \\
        compute_baseline_interactions.py <out_dir>

The second line is this folder's existing scorer, UNCHANGED: the manifest written here is the
layout it already reads, so it fingerprints the new stages without knowing they are new.
"""
import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402

E = "/home1/irteam/VoxBind/voxbind/exps"
POSECHECK = os.path.join(os.path.dirname(HERE), "fig-posecheck")
BUNDLE = next((os.path.join(d, "results/task2-drugdesign")
               for d in (HERE, *(os.path.dirname(HERE[:i]) for i in
                                 range(len(HERE), 0, -1) if HERE[i - 1] == os.sep))
               if os.path.isdir(os.path.join(d, "results/task2-drugdesign"))
               and os.path.isdir(os.path.join(d, "voxbind"))), None)

# The crystal ligands and the pocket crops both come from here, as they do for every other
# builder in 260910 -- compute_baseline_interactions.py globs this same root for its crop.
REF_ROOT = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"

# (label, folder, stem) -- the five published baselines out of the results bundle. FuncBind
# is here and NOT in stage_baseline_poses.py's list: the interaction figures never drew it,
# so this is the first time its poses are staged. Its meta and its posecheck export both
# exist, which is all the join needs.
BASELINES = [("AR", "AR", "AR"),
             ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
             ("DiffSBDD", "DiffSBDD", "DiffSBDD"),
             ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
             ("FuncBind", "FuncBind", "FuncBind")]

# (label, run root) -- the three we run locally. TargetDiff's poses are taken from the
# posecheck_full tree and not from base_drug/eval/targetdiff, which holds only a crystal
# ligand SDF and a metrics.json: metrics.json stores numbers, never conformers.
LOCAL = [("TargetDiff", f"{E}/frozenenc_probes/posecheck_full/targetdiff"),
         ("VoxBind", f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
         ("CoDE", REF_ROOT)]

# "asgen" AND NOT "gen", WHICH WOULD DESTROY DATA. compute_baseline_interactions.py writes
# its output as interactions_<stage key>.json INTO ITS OWN SOURCE FOLDER, so the stage names
# chosen here become filenames there. dock_core_poses.py already owns interactions_CoDE-gen
# .json, interactions_VoxBind-gen.json and interactions_Reference-gen.json -- the 13-per-pocket
# subsample that fig-interaction-pair reads against its redocked columns. A stage called "gen"
# here would overwrite all three with a different, full-scope molecule set, and the damage
# would surface only later, when that figure's "the two columns must be the same molecules"
# check failed a long way from the cause.
STAGES = ("asgen", "relaxfree", "relaxpocket")


def pocket_pdb(target):
    hits = glob.glob(os.path.join(REF_ROOT, target, "*_pocket10.pdb"))
    if not hits:
        raise SystemExit(f"no pocket10 pdb for {target}")
    return hits[0]


def ligand_sdf(target):
    """The crystal ligand: the one .sdf in the target dir that is not samples.sdf."""
    hits = [p for p in glob.glob(os.path.join(REF_ROOT, target, "*.sdf"))
            if os.path.basename(p) != "samples.sdf"]
    if len(hits) != 1:
        raise SystemExit(f"expected one crystal ligand sdf in {target}, found {len(hits)}")
    return hits[0]


def load_meta(folder, stem):
    """export_posecheck_json.py's loader: base, then `_part2` per pocket. Not `_gap.pt`."""
    d = os.path.join(BUNDLE, folder, "samples/meta")
    meta = torch.load(os.path.join(d, f"{stem}.pt"), weights_only=False)
    p2 = os.path.join(d, f"{stem}_part2.pt")
    if os.path.exists(p2):
        meta = [a + b for a, b in zip(meta, torch.load(p2, weights_only=False))]
    return meta


def baseline_mols(label, folder, stem, keep):
    """{pocket: [mol]} for one baseline, in the order its posecheck export scored them.

    THE ORDER IS THE JOIN AND IT IS CHECKED, exactly as stage_baseline_poses.py checks it:
    the heavy-atom sequence of what we stage must equal the export's, or the two are not the
    same molecules and every downstream number is a different population."""
    meta = load_meta(folder, stem)
    export = json.load(open(os.path.join(POSECHECK, f"posecheck_{label}.json")))
    by_pocket = collections.defaultdict(list)
    for m in export["molecules"]:
        by_pocket[m["p"]].append(m)
    out = {}
    for p in keep:
        entries = [e for e in meta[p] if e.get("mol") is not None]
        scored = by_pocket[p]
        entries = entries[:len(scored)]
        ns = [int(e["mol"].GetNumAtoms()) for e in entries]
        if ns != [m["n"] for m in scored]:
            raise SystemExit(f"{label}: heavy-atom sequence differs at pocket {p}")
        out[p] = [Chem.Mol(e["mol"]) for e in entries]
    return out


def local_mols(label, root, keep):
    """{pocket: [mol]} for one locally-run arm, or the crystal ligand for Reference.

    Fragmented samples are dropped -- a molecule written as two disconnected pieces has no
    single pose to relax, and `pick()` in dock_core_poses.py drops them for the same reason."""
    out = {}
    for p in keep:
        target = f"target_{p:02d}"
        path = (ligand_sdf(target) if label == "Reference"
                else os.path.join(root, target, "samples.sdf"))
        if not os.path.exists(path):
            out[p] = []
            continue
        mols = [m for m in Chem.SDMolSupplier(path, sanitize=True) if m is not None]
        out[p] = [Chem.RemoveHs(m) for m in mols if "." not in Chem.MolToSmiles(m)]
    return out


def write_stage(path, rows_mols):
    """One SDF from (row, mol) pairs, each record tagged with its ORIGINAL row.

    The tag is the molecule's index in the generated file, not its position in this one. A
    molecule whose relaxation fails leaves EVERY stage, so positions shift between the file
    and the manifest and only the tag still says which molecule a record is."""
    w = Chem.SDWriter(path)
    for row, mol in rows_mols:
        m = Chem.Mol(mol)
        m.SetProp("row", str(row))
        w.write(m)
    w.close()


def relax_free(mol):
    """PoseCheck's relax: the ligand alone, the protein unseen."""
    mh = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    AllChem.UFFOptimizeMolecule(mh, maxIters=200)
    return Chem.RemoveHs(mh)


def relax_pocket(mol, prot):
    """The ligand minimised with every protein atom pinned.

    `ignoreInterfragInteractions=False` is the whole point: without it the two fragments do
    not see each other and this is just a slower relax_free."""
    mh = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    combo = Chem.CombineMols(prot, mh)
    # FastFindRings, NOT SanitizeMol. UFF typing needs RingInfo initialised and a bare
    # CombineMols leaves it unset -- but a full sanitize RE-PERCEIVES the combined molecule,
    # and on a malformed sulfonyl (an S carrying two single-bonded oxygens and no formal
    # charge, which the generators emit and RDKit accepts) that re-perception leaves an S
    # that UFF cannot type. The molecule was then dropped, and the loss was not random: every
    # one of them was a sulfonyl, so the drop rate tracked how much malformed sulfur a method
    # generates -- a method-correlated bias inside a figure that compares methods.
    #
    # This is a NARROWER fix, not a looser one. Both fragments are sanitised before they are
    # combined, so aromaticity is already on the atoms and survives the combine (30 aromatic
    # atoms either way) and only the ring bookkeeping is absent. Measured on the smoke set:
    # full sanitize types 0 of those 20 molecules and FastFindRings types 20 of 20, while on
    # the molecules where BOTH work the two produce bit-identical poses (RMSD 0.0000 A over
    # 15 molecules). The drop below is now a genuine safety net rather than a routine 5 %.
    Chem.FastFindRings(combo)
    if not AllChem.UFFHasAllMoleculeParams(combo):
        return None
    ff = AllChem.UFFGetMoleculeForceField(combo, ignoreInterfragInteractions=False)
    for i in range(prot.GetNumAtoms()):
        ff.AddFixedPoint(i)
    ff.Initialize()
    ff.Minimize(maxIts=200)
    pos = combo.GetConformer().GetPositions()[prot.GetNumAtoms():]
    out = Chem.Mol(mh)
    conf = out.GetConformer()
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*xyz))
    return Chem.RemoveHs(out)


def relax_job(job):
    """Both relaxations for one (label, pocket). Reads the staged -gen file, writes two.

    A MOLECULE WHOSE RELAXATION FAILS LEAVES ALL THREE STAGES, and the reason travels with
    it. The first draft of this substituted the generated pose for a failed relaxation, which
    is the worst of the options available: the row survives, the column reads "relaxed", and
    the value in it is the generated pose. That biases the comparison toward "relaxation
    changes nothing" precisely on the molecules where it would have changed most, and it is
    invisible downstream because a molecule that genuinely did not move is written the same
    way. Dropping is the rule dock_core_poses.py already uses for a failed dock, for the same
    reason: a difference between the columns has to be the pose and nothing else."""
    out, label, p = job
    target = f"target_{p:02d}"
    src = os.path.join(out, f"{label}-{STAGES[0]}", f"{target}.sdf")
    if not os.path.getsize(src):
        for stage in ("relaxfree", "relaxpocket"):
            open(os.path.join(out, f"{label}-{stage}", f"{target}.sdf"), "w").close()
        return label, p, [], {}
    mols = [m for m in Chem.SDMolSupplier(src, removeHs=False) if m is not None]
    prot = Chem.MolFromPDBFile(pocket_pdb(target), sanitize=False, removeHs=False)
    Chem.SanitizeMol(prot)

    free, pocketed, kept, why = [], [], [], {}
    for mol in mols:
        row = int(mol.GetProp("row"))
        try:
            f = relax_free(mol)
        except Exception as e:
            why[row] = f"relaxfree:{type(e).__name__}"
            continue
        try:
            k = relax_pocket(mol, prot)
        except Exception as e:
            why[row] = f"relaxpocket:{type(e).__name__}"
            continue
        if k is None:
            why[row] = "relaxpocket:uff-typing"
            continue
        free.append((row, f))
        pocketed.append((row, k))
        kept.append(row)
    write_stage(os.path.join(out, f"{label}-relaxfree", f"{target}.sdf"), free)
    write_stage(os.path.join(out, f"{label}-relaxpocket", f"{target}.sdf"), pocketed)
    return label, p, kept, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=28)
    # Both exist for the smoke run. This is a ~1.2 h job over 59k molecules and finding a
    # typo at minute 55 is not a plan; `--limit 2 --only CoDE,FuncBind` finishes in seconds
    # and exercises one local arm and one baseline, which are the two loading paths.
    ap.add_argument("--limit", type=int, default=0, help="use only the first N pockets")
    ap.add_argument("--only", default="", help="comma-separated labels to stage")
    args = ap.parse_args()

    keep = sorted(int(t.split("_")[1]) for t in pc.P79)
    if args.limit:
        keep = keep[:args.limit]
    want = {s for s in args.only.split(",") if s}
    t0 = time.time()

    # ── phase 1: the generated poses and the manifest, in the parent ──────────
    series = [("Reference", None)] + LOCAL
    mols_by = {}
    for label, root in series:
        if want and label not in want:
            continue
        mols_by[label] = local_mols(label, root, keep)
    for label, folder, stem in BASELINES:
        if want and label not in want:
            continue
        mols_by[label] = baseline_mols(label, folder, stem, keep)

    heavy_by = {}                      # (label, pocket) -> heavy-atom count per original row
    manifest = {"pockets": keep, "methods": {},
                "relaxations": {
                    "relaxfree": "AddHs(addCoords) -> UFFOptimizeMolecule(maxIters=200) "
                                 "-> RemoveHs; the protein is not seen",
                    "relaxpocket": "UFF over ligand + pocket10 crop with every protein atom "
                                   "fixed, ignoreInterfragInteractions=False, maxIters=200"},
                "receptor_scope": "pocket10 crop, untrimmed — this folder's convention",
                "note": "gen, relaxfree and relaxpocket are the same molecules row for row"}
    for label in mols_by:
        for stage in STAGES:
            os.makedirs(os.path.join(args.out, f"{label}-{stage}"), exist_ok=True)
        written = 0
        for p in keep:
            mols = mols_by[label][p]
            write_stage(os.path.join(args.out, f"{label}-{STAGES[0]}", f"target_{p:02d}.sdf"),
                        list(enumerate(mols)))
            heavy_by[label, p] = [int(m.GetNumHeavyAtoms()) for m in mols]
            written += len(mols)
        print(f"  staged {label:12s} {written:6d} molecules over {len(keep)} pockets",
              flush=True)
    total = sum(len(v) for v in heavy_by.values())
    print(f"\nphase 1: {total} molecules staged in {time.time() - t0:.0f}s\n"
          f"phase 2: relaxing both ways on {args.workers} workers "
          f"(~{total * 1.95 / args.workers / 60:.0f} min)\n", flush=True)

    # ── phase 2: the two relaxations, in a pool ───────────────────────────────
    jobs = [(args.out, label, p) for label in mols_by for p in keep]
    kept_by, dropped = {}, {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for k, (label, p, kept, why) in enumerate(pool.imap_unordered(relax_job, jobs), 1):
            kept_by[label, p] = kept
            done += len(kept)
            if why:
                dropped[f"{label}-target_{p:02d}"] = why
            if k % 40 == 0 or k == len(jobs):
                print(f"  {k:4d}/{len(jobs)} jobs · {done:6d} molecules · "
                      f"{time.time() - t0:6.0f}s", flush=True)

    # ── phase 3: re-cut the generated stage to the survivors, then the manifest ───
    # The generated file was written before anything was relaxed, so it still holds the
    # molecules the two relaxed stages dropped. Left alone it would give the generated
    # column more molecules than the others -- the population difference this whole
    # arrangement exists to prevent.
    n_drop = 0
    for label in mols_by:
        per_pocket, written = {}, 0
        for p in keep:
            kept = kept_by[label, p]
            path = os.path.join(args.out, f"{label}-{STAGES[0]}", f"target_{p:02d}.sdf")
            mols = {int(m.GetProp("row")): m
                    for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None}
            n_drop += len(mols) - len(kept)
            write_stage(path, [(r, mols[r]) for r in kept if r in mols])
            per_pocket[str(p)] = {"n_rows": len(kept), "rows": kept,
                                  "heavy_atoms": [heavy_by[label, p][r] for r in kept]}
            written += len(kept)
        # Every stage is now the same molecules, row for row -- that identity is what lets
        # the figure read a difference between columns as a difference in pose.
        for stage in STAGES:
            manifest["methods"][f"{label}-{stage}"] = {"n_molecules": written,
                                                       "pockets": per_pocket}
    manifest["dropped"] = dropped
    manifest["note"] += ("; a molecule whose relaxation failed was dropped from all three "
                         "stages and its reason recorded in `dropped`")
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"))
    print(f"\n{done} molecules relaxed both ways, {n_drop} dropped from all three stages "
          f"({100 * n_drop / max(total, 1):.2f}%), in {time.time() - t0:.0f}s\n"
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
