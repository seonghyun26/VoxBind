#!/usr/bin/env python3
"""stage_baseline_samples.py — lay the five published baselines out as sample dirs.

The baselines ship as TargetDiff-format meta bundles (`results/task2-drugdesign/<M>/
samples/meta/<M>{,_part2,_gap}.pt`: one list per test pocket of
{mol, smiles, ligand_filename, pred_pos}). Everything downstream of ours -- metrics.py,
the PoseBusters and PoseCheck builders, the SuCOS script -- keys off a directory holding
`samples.sdf` plus that pocket's `*_pocket10.pdb`. So instead of teaching each of those
about a second input format, this writes the bundles out in the shape they already read:

    voxbind/exps/baselines_pose/<key>/target_XX/
        samples.sdf                 the pocket's generated molecules
        <receptor>_pocket10.pdb     hard link from the vanilla run
        <ligand>.sdf                hard link, the crystal reference

Hard links, not copies: same filesystem, and the receptor a baseline is scored against is
then provably the same file our own arms are scored against, not a copy that could drift.

THE POCKET MAPPING IS CHECKED, NOT ASSUMED. Bundle index i is claimed to be
`target_{i:02d}`, and this verifies it for every pocket by comparing the entry's
`ligand_filename` against the crystal-ligand SDF sitting in that target dir (their only
difference is that ours joins the subdirectory with `__` instead of `/`). A silent
index scramble here would produce molecules scored against the wrong pocket and numbers
that look entirely reasonable, so a mismatch aborts.

ONLY THE 79 ELECTRON-DENSITY POCKETS are staged: every figure is drawn over those, and the
other 21 would be a fifth more PoseBusters compute that nothing reads.

    /opt/conda/envs/voxbind/bin/python voxbind/scripts/tools/stage_baseline_samples.py
"""
import argparse
import glob
import json
import os

import torch
from rdkit import Chem, RDLogger

ROOT = "/home1/irteam/VoxBind"
BUNDLES = os.path.join(ROOT, "results/task2-drugdesign")
VANILLA = os.path.join(ROOT, "voxbind/exps/_vanilla_ep923/samples/full_eval_ep923")
OUT_ROOT = os.path.join(ROOT, "voxbind/exps/baselines_pose")
P79 = json.load(open(os.path.join(
    ROOT, "voxbind/exps/frozenenc_probes/p79_targets.json")))

# key -> (folder, bundle stem). DecompDiff's bundle carries the reference-prior variant,
# which is the one the paper tables and every other figure in this project use.
METHODS = [
    ("ar",         "AR",         "AR"),
    ("pocket2mol", "Pocket2Mol", "Pocket2Mol"),
    ("diffsbdd",   "DiffSBDD",   "DiffSBDD"),
    ("decompdiff", "DecompDiff", "DecompDiff_ref_prior"),
    ("funcbind",   "FuncBind",   "FuncBind"),
]
PARTS = ("", "_part2", "_gap")     # main, second half, re-run fill; disjoint by SMILES


def load_bundle(folder, stem):
    """[pocket index] -> [entry], the three parts concatenated in order."""
    merged = [[] for _ in range(100)]
    for suffix in PARTS:
        path = os.path.join(BUNDLES, folder, "samples", "meta", f"{stem}{suffix}.pt")
        if not os.path.exists(path):
            continue
        for i, entries in enumerate(torch.load(path, weights_only=False)):
            merged[i].extend(entries)
    return merged


def reference_sdf(target):
    hits = [p for p in glob.glob(os.path.join(VANILLA, target, "*.sdf"))
            if os.path.basename(p) != "samples.sdf"]
    return hits[0] if hits else None


def pocket_pdb(target):
    hits = glob.glob(os.path.join(VANILLA, target, "*_pocket10.pdb"))
    return hits[0] if hits else None


def link(src, dst):
    if src and not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:                      # cross-device or already there
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rewrite samples.sdf even where it already exists")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")

    for key, folder, stem in METHODS:
        bundle = load_bundle(folder, stem)
        written = mols_out = mismatched = 0
        for target in P79:
            i = int(target.split("_")[1])
            ref, pdb = reference_sdf(target), pocket_pdb(target)
            entries = bundle[i]
            if not entries:
                continue
            # the check that makes the index trustworthy
            want = entries[0]["ligand_filename"].replace("/", "__")
            if ref is None or os.path.basename(ref) != want:
                mismatched += 1
                print(f"  MISMATCH {key} {target}: bundle says {want}, "
                      f"target holds {os.path.basename(ref) if ref else 'nothing'}")
                continue
            d = os.path.join(OUT_ROOT, key, target)
            os.makedirs(d, exist_ok=True)
            link(pdb, os.path.join(d, os.path.basename(pdb))) if pdb else None
            link(ref, os.path.join(d, os.path.basename(ref)))
            sdf = os.path.join(d, "samples.sdf")
            if os.path.exists(sdf) and not args.force:
                continue
            w = Chem.SDWriter(sdf)
            n = 0
            for e in entries:
                m = e.get("mol")
                if m is None:
                    continue
                try:
                    w.write(m)
                    n += 1
                except Exception:            # noqa: BLE001 — an unwritable mol is data
                    continue
            w.close()
            written += 1
            mols_out += n
        if mismatched:
            raise SystemExit(f"{key}: {mismatched} pocket(s) failed the index check — "
                             f"refusing to stage a possibly scrambled set")
        print(f"{folder:12s} {written:3d} pockets, {mols_out:6d} molecules "
              f"-> {os.path.join(OUT_ROOT, key)}")


if __name__ == "__main__":
    main()
