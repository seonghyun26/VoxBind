#!/usr/bin/env python3
"""stage_baseline_poses.py — step 1 of 2: baseline poses out of the results bundle, as SDF.

    <out>/<Method>/target_NN.sdf   the p79 poses, in posecheck_<Method>.json order
    <out>/manifest.json            per pocket: the heavy-atom sequence, for re-checking

WHY TWO STEPS. The molecules live in `results/task2-drugdesign/<M>/samples/meta/*.pt`,
which only `torch` can open, and `torch` is in the `voxbind` env. PoseCheck and ProLIF are
in `moleval`, which has no torch. Neither env can do both halves, so this one writes SDF
and `compute_baseline_interactions.py` reads it back under the other interpreter.

An SDF round trip is not a compromise here: the three local arms are themselves read from
`samples.sdf` by every builder in 260910, so putting the baselines through the same
serialisation makes them MORE comparable, not less.

THE ORDER IS THE JOIN AND IT IS CHECKED TWICE. Molecules are written in exactly the order
`export_posecheck_json.py` walked them (meta base + `_part2` per pocket, `mol is None`
skipped, truncated to what the export actually scored), the heavy-atom sequence is verified
against the export here, and each record carries its position in an SD tag `row` so step 2
stays aligned even if RDKit refuses to read one back.

    /opt/conda/envs/voxbind/bin/python stage_baseline_poses.py <out_dir>
"""
import collections
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
POSECHECK = os.path.join(os.path.dirname(HERE), "fig-posecheck")
sys.path.insert(0, os.path.dirname(HERE))
import pose_common as pc                                             # noqa: E402

BUNDLE = next((os.path.join(d, "results/task2-drugdesign")
               for d in (HERE, *(os.path.dirname(HERE[:i]) for i in
                                 range(len(HERE), 0, -1) if HERE[i - 1] == os.sep))
               if os.path.isdir(os.path.join(d, "results/task2-drugdesign"))
               and os.path.isdir(os.path.join(d, "voxbind"))), None)

BASELINES = [("AR", "AR", "AR"), ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
             ("DiffSBDD", "DiffSBDD", "DiffSBDD"),
             ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior")]


def load_meta(folder, stem):
    """export_posecheck_json.py's loader: base, then `_part2` per pocket. Not `_gap.pt`."""
    d = os.path.join(BUNDLE, folder, "samples/meta")
    meta = torch.load(os.path.join(d, f"{stem}.pt"), weights_only=False)
    p2 = os.path.join(d, f"{stem}_part2.pt")
    if os.path.exists(p2):
        meta = [a + b for a, b in zip(meta, torch.load(p2, weights_only=False))]
    return meta


def main():
    out = sys.argv[1]
    keep = sorted(int(t.split("_")[1]) for t in pc.P79)
    manifest = {"pockets": keep, "methods": {}}
    for label, folder, stem in BASELINES:
        meta = load_meta(folder, stem)
        export = json.load(open(os.path.join(POSECHECK, f"posecheck_{label}.json")))
        by_pocket = collections.defaultdict(list)
        for m in export["molecules"]:
            by_pocket[m["p"]].append(m)

        d = os.path.join(out, label)
        os.makedirs(d, exist_ok=True)
        per_pocket, written = {}, 0
        for p in keep:
            entries = [e for e in meta[p] if e.get("mol") is not None]
            scored = by_pocket[p]
            entries = entries[:len(scored)]
            ns = [int(e["mol"].GetNumAtoms()) for e in entries]
            if ns != [m["n"] for m in scored]:
                raise SystemExit(f"{label}: heavy-atom sequence differs at pocket {p}")
            w = Chem.SDWriter(os.path.join(d, f"target_{p:02d}.sdf"))
            for row, e in enumerate(entries):
                mol = Chem.Mol(e["mol"])
                mol.SetProp("row", str(row))
                w.write(mol)
                written += 1
            w.close()
            per_pocket[str(p)] = {"n_rows": len(entries), "heavy_atoms": ns}
        manifest["methods"][label] = {"n_molecules": written, "pockets": per_pocket}
        print(f"{label:12s} {written:6d} molecules over {len(keep)} pockets -> {d}")
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))
    print(f"\nmanifest -> {os.path.join(out, 'manifest.json')}")


if __name__ == "__main__":
    main()
