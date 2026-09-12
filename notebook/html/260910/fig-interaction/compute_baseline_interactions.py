#!/usr/bin/env python3
"""compute_baseline_interactions.py — step 2 of 2: PoseCheck interactions for the baselines.

    interactions_<Method>.json   per molecule: pocket, heavy atoms, interaction counts

Reads what `stage_baseline_poses.py` wrote and scores it with PoseCheck/ProLIF against the
SAME pocket10 crop the three local arms were scored against, so every series in
`build_interactions.py` comes from one protocol.

WHY THIS CAN BE RECOMPUTED AT ALL. `posecheck_<Method>.json` carries only strain and
clashes -- the fingerprint was never exported, and the raw per-chunk PoseCheck output stays
on svr12. But the poses themselves are now in the results bundle, so the fingerprint can be
recomputed from them rather than fetched.

AND WHY RECOMPUTING IS SAFE. The local arms' fingerprints in their `metrics.json` were
written by a different environment from this one, so agreeing with them is a requirement,
not an assumption: re-scoring 298 molecules of `voxbind_frozenenc_atomblob7_v2p1_sig0.9`
over three pockets with THIS interpreter reproduced the stored counts exactly -- 100.0 % on
all four types, 298/298 each. `--verify` re-runs that check.

THE RECEPTOR IS THE pocket10 CROP, one per pocket, loaded ONCE and reused across the four
methods -- protonation is the expensive step (~1.7 s) and it does not depend on the ligand.
The baselines' published strain/clash numbers were whole-receptor; these interaction numbers
are not, deliberately, because the figure compares them against local arms and crystal
ligands that are all crop-scored. Measured cost of the crop on the same molecules: 1.8-3.2 %
fewer VdW contacts, within 0.03 on the other three types.

    PATH=/opt/conda/envs/moleval/bin:$PATH \\
    /opt/conda/envs/moleval/bin/python compute_baseline_interactions.py <staged_dir>
    ... --verify        # only re-run the agreement check against the local arms
"""
import collections
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from posecheck import PoseCheck

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
E = "/home1/irteam/VoxBind/voxbind/exps"
REF_ROOT = f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
TYPES = ["VdWContact", "Hydrophobic", "HBAcceptor", "HBDonor"]
VERIFY_TARGETS = ("target_02", "target_03", "target_05")


def pocket_pdb(target):
    hits = glob.glob(os.path.join(REF_ROOT, target, "*_pocket10.pdb"))
    if not hits:
        raise SystemExit(f"no pocket10 pdb for {target}")
    return hits[0]


def per_molecule(pc_obj, mols):
    """[{type: count}] aligned to `mols`. A type a molecule does not make is simply absent
    from its dict -- the same convention metrics.json uses, which pose_common's loader and
    build_interactions.py's `counts()` both already depend on."""
    pc_obj.load_ligands_from_mols(mols)
    df = pc_obj.calculate_interactions()
    out = []
    for row in range(len(mols)):
        got = collections.Counter()
        for col in df.columns:
            v = df.iloc[row][col]
            if v:
                got[col[2]] += int(v)
        out.append({k: int(v) for k, v in got.items()})
    return out


def verify():
    """Re-score local-arm molecules whose fingerprints are already stored, and require this
    interpreter to reproduce them."""
    tot, agree = collections.Counter(), collections.Counter()
    n = 0
    for target in VERIFY_TARGETS:
        root = os.path.join(REF_ROOT, target)
        j = json.load(open(os.path.join(root, "metrics.json")))
        mols = list(Chem.SDMolSupplier(os.path.join(root, "samples.sdf"), removeHs=False))
        keep = [(i, m) for i, m in enumerate(mols) if m is not None]
        pc_obj = PoseCheck()
        pc_obj.load_protein_from_pdb(pocket_pdb(target))
        got = per_molecule(pc_obj, [m for _, m in keep])
        for row, (i, _) in enumerate(keep):
            ref = (j["samples"][row].get("posecheck") or {}).get("interactions") or {}
            n += 1
            for t in TYPES:
                tot[t] += 1
                agree[t] += got[row].get(t, 0) == ref.get(t, 0)
    print(f"verify: {n} local-arm molecules over {len(VERIFY_TARGETS)} pockets")
    ok = True
    for t in TYPES:
        pct = 100 * agree[t] / tot[t]
        ok &= agree[t] == tot[t]
        print(f"  {t:12s} {pct:6.2f}%  ({agree[t]}/{tot[t]})")
    if not ok:
        raise SystemExit("this interpreter does not reproduce the stored fingerprints; "
                         "the baselines must not be scored against them")
    print("  -> reproduces the stored fingerprints exactly")


def main():
    if "--verify" in sys.argv:
        verify()
        return
    staged = sys.argv[1]
    manifest = json.load(open(os.path.join(staged, "manifest.json")))
    methods = list(manifest["methods"])
    pockets = manifest["pockets"]
    verify()

    rows = {m: [] for m in methods}
    t0 = time.time()
    for k, p in enumerate(pockets):
        target = f"target_{p:02d}"
        pc_obj = PoseCheck()
        pc_obj.load_protein_from_pdb(pocket_pdb(target))     # once per pocket, not per method
        for m in methods:
            path = os.path.join(staged, m, f"{target}.sdf")
            entry = manifest["methods"][m]["pockets"][str(p)]
            heavy = entry["heavy_atoms"]
            # A staged set that dropped molecules lists the surviving row indices, which are
            # then discontiguous; one that dropped none just numbers them 0..n-1. Walking
            # `rows` beside `heavy_atoms` keeps a molecule's heavy-atom count with the
            # molecule rather than with its position.
            idx = entry.get("rows", list(range(entry["n_rows"])))
            # A pocket that lost its whole column -- target_71's receptor cannot be
            # protonated, so nothing docked there -- leaves a zero-byte file, and
            # SDMolSupplier raises on one instead of reading no molecules.
            read = [] if not os.path.getsize(path) else [
                (int(x.GetProp("row")), x)
                for x in Chem.SDMolSupplier(path, removeHs=False) if x is not None]
            got = per_molecule(pc_obj, [x for _, x in read]) if read else []
            ifp = {row: g for (row, _), g in zip(read, got)}
            for row, n in zip(idx, heavy):
                rows[m].append({"p": p, "n": n, "ifp": ifp.get(row)})
        if (k + 1) % 20 == 0 or k + 1 == len(pockets):
            print(f"  {k + 1:3d}/{len(pockets)} pockets · {time.time() - t0:6.1f}s",
                  flush=True)

    for m in methods:
        got = sum(r["ifp"] is not None for r in rows[m])
        doc = {
            "method": m,
            "metric": "PoseCheck / ProLIF interaction fingerprint (Harris et al., 2023)",
            "receptor_scope": "pocket10 crop, the same one the local arms are scored on",
            "source": f"results/task2-drugdesign/{m}/samples/meta, staged by "
                      f"stage_baseline_poses.py and re-scored here",
            "verified": "this interpreter reproduces the local arms' stored fingerprints "
                        "exactly (298/298 molecules, all four types)",
            "zero_handling": "a type absent from a molecule's dict was not made; count it 0",
            "n_pockets": len(pockets), "n_molecules": len(rows[m]), "n_scored": got,
            "types": TYPES, "molecules": rows[m],
        }
        out = os.path.join(HERE, f"interactions_{m}.json")
        json.dump(doc, open(out, "w"), separators=(",", ":"))
        print(f"{m:12s} {len(rows[m]):6d} molecules, {got:6d} scored -> "
              f"{os.path.basename(out)}")


if __name__ == "__main__":
    main()
