#!/usr/bin/env python3
"""dock_core_poses.py — re-dock the core arms and stage BOTH poses for fingerprinting.

    <out>/manifest.json
    <out>/<Arm>-gen/target_XX.sdf        the pose as generated
    <out>/<Arm>-docked/target_XX.sdf     the same molecule, re-docked

VoxBind's Fig. 14 draws each interaction type twice: the pose as generated, and the same
molecule redocked. This folder has only the first, because nothing stores the second --
`metrics.py`'s `run_vina_docking` calls Vina and keeps `{"score_only", "minimize", "dock"}`,
three floats, while the docked conformer it just computed is dropped on the floor. So the
second series has to be produced, and that means docking again.

THE DOCKED POSE IS THE SAME MOLECULE, MOVED, and this script is built so that that is true
by construction rather than by hope. Vina hands back the pose as PDBQT. The obvious route --
openbabel PDBQT -> SDF, then AssignBondOrdersFromTemplate -- was tried and REJECTED: openbabel
fails to kekulize the aromatic rings, and the molecule that comes out the other side has a
different SMILES from the one that went in. Fingerprinting that would be scoring a different
chemistry and calling it a redocked pose. Instead only COORDINATES cross the boundary:

  * the input .pdbqt carries the template's own coordinates -- preparation adds hydrogens
    with addCoords=True and moves no heavy atom -- so matching its ATOM records to template
    atoms by position is exact and unambiguous, and the script aborts if any atom fails to
    match or two match the same one;
  * Vina preserves atom order between the input ligand and the output pose, so that same
    permutation reads the docked coordinates;
  * the molecule written out is `Chem.Mol(template)` with a moved conformer. Nothing
    re-perceives an element, a bond or a charge.

Validated before this script existed: three molecules of target_02 round-tripped with
identical SMILES, every heavy atom mapped, centroids moved 0.5-2.0 A.

PAIRED, NOT POOLED. Both series cover exactly the same molecules, so a difference between
them is the docking and nothing else -- not a difference in which molecules were sampled.
That is why `-gen` is staged here too rather than read from the existing
interactions.json: one scorer, one molecule set, two poses. A dock that fails therefore
takes its molecule out of BOTH series and its reason into `manifest.json["dropped"]`;
keeping the generated pose of a molecule that has no docked pose would put the gap right
back into the comparison. Two things fail deterministically and neither is a defect here:
target_71, whose pocket10 receptor has a chain gap at GLU 866 that pdb2pqr30 refuses, so
it loses all three arms and the paired figure rests on 78 pockets like every other docking
result in this project; and the occasional degenerate sample (VoxBind target_96 row 8 is
`CNNO`, four heavy atoms), whose extent gives Vina a box with a zero dimension.

THE SUBSAMPLE. Letter values down to p1.6 are stable at about a thousand molecules, so the
model arms take `--per-pocket` (13 by default, evenly spaced across generation order, NEVER
a prefix -- the sampling order is not random for every arm) and the reference takes its one
crystal ligand per pocket. That is ~2,100 docks instead of ~15,800, and the figure it feeds
is indistinguishable. Pass `--per-pocket 0` for every molecule.

EVERY WORKER NEEDS ITS OWN DIRECTORY. TargetDiff's PrepLig writes `tmp_h.sdf` and
`conf_h.sdf` into the CURRENT WORKING DIRECTORY, unconditionally and by that fixed name, so
a pool of workers sharing a cwd would have them writing over each other.

    /opt/conda/envs/voxdock/bin/python dock_core_poses.py <scratch>/core_poses [--workers 28]
    PATH=/opt/conda/envs/moleval/bin:$PATH /opt/conda/envs/moleval/bin/python \\
        compute_baseline_interactions.py <scratch>/core_poses

The second line is this folder's existing scorer, unchanged: the staged layout here is the
one `stage_baseline_poses.py` writes, so the same verified interpreter scores both.
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
# vina 1.2.2 calls np.int / np.bool, removed in numpy 1.24. Restore before it is imported.
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

sys.path.insert(0, "/home1/irteam/TargetDIff")

from rdkit import Chem, RDLogger                                     # noqa: E402
from rdkit.Geometry import Point3D                                   # noqa: E402

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"

# label -> run root. The core three, in the drug-design table's order. The label is what
# ends up in interactions_<label>-{gen,docked}.json, so it is spelled the way the exports
# spell a method, not the way a figure prints one.
ARMS = [
    ("Reference", f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
    ("VoxBind",   f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",      f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]
REF_ROOT = ARMS[0][1]
P79 = json.load(open(f"{E}/frozenenc_probes/p79_targets.json"))
EXHAUSTIVENESS = 16               # the protocol the Vina table was produced with
ATOM = re.compile(r"^(?:ATOM|HETATM)")


def pocket_pdb(target):
    hits = glob.glob(os.path.join(REF_ROOT, target, "*_pocket10.pdb"))
    if not hits:
        raise SystemExit(f"no pocket10 pdb for {target}")
    return hits[0]


def ligand_sdf(root, target):
    """The crystal ligand: the one .sdf in the target dir that is not samples.sdf."""
    hits = [p for p in glob.glob(os.path.join(root, target, "*.sdf"))
            if os.path.basename(p) != "samples.sdf"]
    if len(hits) != 1:
        raise SystemExit(f"expected one crystal ligand sdf in {target}, found {len(hits)}")
    return hits[0]


def pick(root, target, label, per_pocket):
    """The molecules to dock for one arm and pocket, as RDKit mols without hydrogens.

    Evenly spaced across the file, never the leading `n`: generation order is not random
    for every arm, and a prefix would be a different population from the tail. Same rule
    the Vina tables use for their 100-per-pocket draw."""
    if label == "Reference":
        path = ligand_sdf(root, target)
    else:
        path = os.path.join(root, target, "samples.sdf")
    if not os.path.exists(path):
        return []
    mols = [m for m in Chem.SDMolSupplier(path, sanitize=True) if m is not None]
    mols = [m for m in mols if "." not in Chem.MolToSmiles(m)]
    if per_pocket and len(mols) > per_pocket:
        idx = np.linspace(0, len(mols) - 1, per_pocket).round().astype(int)
        mols = [mols[i] for i in sorted(set(idx.tolist()))]
    return [Chem.RemoveHs(m) for m in mols]


def pdbqt_atoms(text):
    """(element, xyz) per ATOM record, in file order."""
    out = []
    for ln in text.splitlines():
        if not ATOM.match(ln):
            continue
        out.append((ln[76:79].strip() or ln[12:16].strip(),
                    (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    return out


def moved_copy(template, lig_pdbqt_text, pose_text):
    """`template` with the docked coordinates on it. Raises rather than returning anything
    approximate -- see the module docstring."""
    src, dst = pdbqt_atoms(lig_pdbqt_text), pdbqt_atoms(pose_text)
    if len(src) != len(dst):
        raise ValueError(f"pdbqt atom count changed: {len(src)} -> {len(dst)}")
    pos = template.GetConformer().GetPositions()
    used, moves = set(), {}
    for i, (_, xyz) in enumerate(src):
        d = np.linalg.norm(pos - np.asarray(xyz), axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-2:                 # a hydrogen preparation added; no heavy atom for it
            continue
        if j in used:
            raise ValueError(f"two pdbqt atoms map onto template atom {j}")
        used.add(j)
        moves[j] = dst[i][1]
    if len(moves) != template.GetNumHeavyAtoms():
        raise ValueError(f"mapped {len(moves)} of {template.GetNumHeavyAtoms()} heavy atoms")
    out = Chem.Mol(template)
    conf = out.GetConformer()
    for j, xyz in moves.items():
        conf.SetAtomPosition(j, Point3D(*xyz))
    return out


def _init():
    """One scratch directory per worker, and become it -- PrepLig writes tmp_h.sdf and
    conf_h.sdf into the cwd by that fixed name."""
    d = os.path.join(_init.tmp_root, f"w{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    os.chdir(d)


def dock_one_target(job):
    """Dock every selected molecule of one (arm, pocket). Returns SDF blocks for both
    series so the parent writes the files and nothing races.

    A MOLECULE WHOSE DOCK FAILS LEAVES BOTH SERIES, not just the docked one. Writing its
    generated pose anyway would put a molecule in one column that the other column cannot
    have, and the difference between the columns is supposed to be the docking alone. The
    reason travels with it so the manifest records why a row is absent instead of leaving
    a silent gap for someone to re-derive later."""
    from copy import deepcopy
    from utils.evaluation.docking_vina import VinaDockingTask

    label, root, target, per_pocket, exh = job
    mols = pick(root, target, label, per_pocket)
    rec, tmp = pocket_pdb(target), os.getcwd()
    gen, docked, heavy, kept, dropped = [], [], [], [], []
    for row, mol in enumerate(mols):
        try:
            task = VinaDockingTask(rec, deepcopy(mol), tmp_dir=tmp)
            r = task.run(mode="dock", exhaustiveness=exh, cpu=1)[0]
            pose = r["pose"][0] if isinstance(r["pose"], (list, tuple)) else r["pose"]
            lig = open(task.ligand_path[:-4] + ".pdbqt").read()
            d = moved_copy(mol, lig, pose)
        except Exception as e:                 # noqa: BLE001 — one bad dock is a gap, not a stop
            dropped.append({"row": row, "smiles": Chem.MolToSmiles(mol),
                            "reason": f"{type(e).__name__}: {e}".splitlines()[0][:200]})
            continue
        d.SetProp("row", str(row))
        d.SetProp("vina_dock", f"{r['affinity']:.3f}")
        m = Chem.Mol(mol)
        m.SetProp("row", str(row))
        gen.append(m)
        docked.append(d)
        heavy.append(mol.GetNumHeavyAtoms())
        kept.append(row)
    return label, target, sdf_text(gen), sdf_text(docked), heavy, kept, len(mols), dropped


def sdf_text(mols):
    """SDF text WITH the SD tags. Chem.MolToMolBlock drops properties -- it writes the
    connection table and nothing else -- and `row` is the tag the scorer keys every
    molecule by, so this must go through SDWriter."""
    buf = io.StringIO()
    w = Chem.SDWriter(buf)
    for m in mols:
        w.write(m)
    w.close()
    return buf.getvalue()


def write_series(out, name, target, text):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{target}.sdf"), "w") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-pocket", type=int, default=13,
                    help="molecules per pocket per model arm; 0 for all")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--exhaustiveness", type=int, default=EXHAUSTIVENESS)
    ap.add_argument("--pockets", type=int, default=0,
                    help="first N pockets only; for smoke tests, 0 for the whole set")
    args = ap.parse_args()
    targets = P79[:args.pockets] if args.pockets else P79

    os.makedirs(args.out, exist_ok=True)
    tmp_root = os.path.join(args.out, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    _init.tmp_root = tmp_root

    jobs = [(lab, root, t, args.per_pocket, args.exhaustiveness)
            for lab, root in ARMS for t in targets]
    print(f"{len(targets)} pockets x {len(ARMS)} arms = {len(jobs)} jobs · "
          f"{args.per_pocket or 'all'} molecules per pocket · "
          f"exhaustiveness {args.exhaustiveness} · {args.workers} workers\n", flush=True)

    per = {lab: {} for lab, _ in ARMS}
    done, fails, t0 = 0, 0, time.time()
    with mp.Pool(args.workers, initializer=_init) as pool:
        for lab, target, gen, docked, heavy, kept, n, drop in pool.imap_unordered(
                dock_one_target, jobs):
            write_series(args.out, f"{lab}-gen", target, gen)
            write_series(args.out, f"{lab}-docked", target, docked)
            per[lab][target] = {"n_rows": len(kept), "n_picked": n, "rows": kept,
                                "heavy_atoms": heavy, "dropped": drop}
            done += 1
            fails += len(drop)
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done:3d}/{len(jobs)} jobs · {el:7.1f}s · "
                      f"eta {el / done * (len(jobs) - done):7.1f}s · {fails} dropped",
                      flush=True)

    pockets = [int(t.split("_")[1]) for t in targets]
    methods = {}
    for lab, _ in ARMS:
        for suffix in ("gen", "docked"):
            # `rows` is the surviving row indices, which a dropped dock makes discontiguous.
            # The scorer walks it alongside heavy_atoms; without it it would number the
            # molecules 0..n-1 and read one pocket's heavy-atom counts off by the drop.
            methods[f"{lab}-{suffix}"] = {
                "pockets": {str(int(t.split("_")[1])): {
                    "n_rows": per[lab][t]["n_rows"],
                    "rows": per[lab][t]["rows"],
                    "heavy_atoms": per[lab][t]["heavy_atoms"]} for t in targets}}
    # Every molecule that left both series, with the reason. A pocket whose receptor cannot
    # be protonated loses its whole column here, so this is the record of which pockets the
    # paired figure actually rests on.
    dropped = {f"{lab}-{t}": per[lab][t]["dropped"]
               for lab, _ in ARMS for t in targets if per[lab][t]["dropped"]}
    json.dump({"pockets": pockets, "methods": methods,
               "per_pocket": args.per_pocket, "exhaustiveness": args.exhaustiveness,
               "picked": sum(per[lab][t]["n_picked"] for lab, _ in ARMS for t in targets),
               "dropped": dropped,
               "note": "gen and docked cover the same molecules, row for row; a dock that "
                       "failed took its molecule out of both"},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

    picked = sum(per[lab][t]["n_picked"] for lab, _ in ARMS for t in targets)
    total = sum(per[lab][t]["n_rows"] for lab, _ in ARMS for t in targets)
    print(f"\n{picked} molecules picked, {total} paired ({fails} dropped) in "
          f"{time.time() - t0:.0f}s\nwrote {args.out}")


if __name__ == "__main__":
    main()
