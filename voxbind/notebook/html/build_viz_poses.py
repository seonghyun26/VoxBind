#!/usr/bin/env python
"""Generate poses_<tag>.json for a viz_target<N>/ page (Vina min/dock poses).

Companion to build_viz_target.py. That builder writes the generated poses +
score numbers; this one re-runs Vina (voxdock env) to produce the *minimized*
and *docked* structures so the viewer's "Vina-min / Vina-dock" pose radios work.

Output schema (matches the hand-made viz_target92):
    poses_<tag>.json = { "<mol_idx>": {min_aff, dock_aff, min_sdf, dock_sdf}, ... }
mol_idx is index into that model's samples_<tag>.sdf (== scores[tag] order), so
it must use the SAME valid-filter/order as build_viz_target.py.

Run with the voxdock env python:
    /opt/conda/envs/voxdock/bin/python notebook/html/build_viz_poses.py 58 74
"""
import os, sys, json, glob, time, argparse

import numpy as np
# vina 1.2.2 calls np.int (removed in numpy>=1.24); restore aliases in-process.
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

sys.path.insert(0, "/home1/irteam/TargetDIff")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from utils.evaluation.docking_vina import VinaDockingTask

try:
    from openbabel import pybel  # openbabel 3
except ImportError:
    import pybel  # openbabel 2

HERE = os.path.dirname(os.path.abspath(__file__))          # voxbind/notebook/html
VOXBIND = os.path.dirname(os.path.dirname(HERE))           # voxbind/
EXH = 16                                                   # match the docking eval

RUNS = [
    ("frozenenc_sig0.9", "exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
    ("vanilla_sig0.9",   "exps/exp_sig0.9_v2/samples/full_eval_ep350"),
    ("vanilla_sig1.0",   "exps/exp_sig1.0_350ep/samples/full_eval_ep349"),
]
BASE_TAG = RUNS[0][0]


def load_run(root):
    d = json.load(open(os.path.join(VOXBIND, root, "eval_docking_results.json")))
    return {t["receptor"]: t for t in d["per_target"]}, {t["target"]: t for t in d["per_target"]}


def valid_mols(sdf):
    """Same valid-filter/order as build_viz_target.py so idx aligns with scores[tag]."""
    out = []
    for m in Chem.SDMolSupplier(sdf, sanitize=True):
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if "." in smi:
            continue
        out.append(m)
    return out


def pdbqt_to_sdf(pose):
    """Vina pose (pdbqt string) -> SDF molblock string via OpenBabel."""
    if not pose:
        return None
    try:
        m = pybel.readstring("pdbqt", pose)
        return m.write("sdf")
    except Exception:
        return None


def gen_poses(sdf, receptor, tmp, tag, tname, only_indices=None, modes=None):
    mols = valid_mols(sdf)
    if only_indices is None:
        only_indices = set(range(len(mols)))
    else:
        only_indices = set(only_indices)
    if modes is None:
        modes = {"minimize", "dock"}
    out = {}
    t_start = time.time()
    n_done = 0
    for i, mol in enumerate(mols):
        if i not in only_indices:
            continue
        rec = {"min_aff": None, "dock_aff": None, "min_sdf": None, "dock_sdf": None}
        for mode, aff_key, sdf_key in (("minimize", "min_aff", "min_sdf"),
                                       ("dock", "dock_aff", "dock_sdf")):
            if mode not in modes:
                continue
            try:
                task = VinaDockingTask(protein_path=receptor, ligand_rdmol=mol, tmp_dir=tmp)
                r = task.run(mode=mode, exhaustiveness=EXH)
                rec[aff_key] = float(r[0]["affinity"])
                rec[sdf_key] = pdbqt_to_sdf(r[0]["pose"])
            except Exception as e:  # noqa: BLE001
                pass
        out[str(i)] = rec
        n_done += 1
        if n_done % 20 == 0 or n_done == len(only_indices):
            el = time.time() - t_start
            print(f"    {tname} {tag}: {n_done}/{len(only_indices)}  "
                  f"({el:.0f}s, {el/n_done:.1f}s/mol)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="frozen-enc target indices, e.g. 58 74")
    ap.add_argument("--models", default=",".join(t for t, _ in RUNS),
                    help="comma list of tags to (re)pose; default all three")
    ap.add_argument("--indices", default=None,
                    help="comma list of valid-molecule indices; default all")
    ap.add_argument("--modes", default="minimize,dock",
                    help="comma list chosen from minimize,dock; default both")
    ap.add_argument("--force", action="store_true", help="overwrite existing poses_<tag>.json")
    args = ap.parse_args()
    want = set(args.models.split(","))
    only_indices = None if args.indices is None else {int(x) for x in args.indices.split(",")}
    modes = set(args.modes.split(","))
    unknown_modes = modes - {"minimize", "dock"}
    if unknown_modes:
        ap.error(f"unknown --modes value(s): {','.join(sorted(unknown_modes))}")

    runs = {tag: load_run(root) for tag, root in RUNS}
    root_of = dict(RUNS)

    for idx in (int(x) for x in args.targets):
        tname = f"target_{idx:02d}"
        base = runs[BASE_TAG][1].get(tname)
        if base is None:
            print(f"{tname}: not in {BASE_TAG} — skip"); continue
        receptor = base["receptor"]
        outdir = os.path.join(HERE, f"viz_target{idx}")
        if not os.path.isdir(outdir):
            print(f"{tname}: {outdir} missing (run build_viz_target.py first) — skip"); continue
        print(f"== {tname} ({receptor[:38]}) ==", flush=True)
        for tag, root in RUNS:
            if tag not in want:
                continue
            outpath = os.path.join(outdir, f"poses_{tag}.json")
            if os.path.exists(outpath) and not args.force:
                print(f"    {tag}: exists — skip (use --force)"); continue
            t = runs[tag][0].get(receptor)
            if t is None:
                print(f"    {tag}: no matching receptor — skip"); continue
            tdir = os.path.join(VOXBIND, root, t["target"])
            rec_pdb = glob.glob(os.path.join(tdir, "*_pocket10.pdb"))[0]
            tmp = os.path.join(tdir, ".pose_tmp"); os.makedirs(tmp, exist_ok=True)
            poses = gen_poses(os.path.join(tdir, "samples.sdf"), rec_pdb, tmp, tag, tname,
                              only_indices=only_indices, modes=modes)
            json.dump(poses, open(outpath, "w"))
            done = sum(1 for v in poses.values() if v["dock_sdf"])
            print(f"    {tag}: wrote {outpath}  ({done}/{len(poses)} docked poses)", flush=True)
    print("POSES_DONE", flush=True)


if __name__ == "__main__":
    main()
