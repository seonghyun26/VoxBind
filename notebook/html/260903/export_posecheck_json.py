"""export_posecheck_json.py — per-molecule PoseCheck results for the five baselines,
one JSON per method, so the figures can be redrawn on the box that also holds
TargetDiff / vanilla VoxBind / Ours v1.

The figure script needs three numbers per molecule: heavy-atom count, UFF strain and
steric clashes. PoseCheck stored strain and clashes per (pocket, chunk-of-20) without the
atom count, so the counts are read back from the meta the chunks were built from; the
alignment is re-checked on every chunk (matching length AND ligand_filename) and the
export aborts rather than guessing.

All 100 pockets are exported, each molecule tagged with its pocket index, and the 79
electron-density pocket indices are listed separately -- so the receiving side can draw
either subset without another round trip.

    python notebook/html/260903/export_posecheck_json.py
"""
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "posecheck eval")
BASE = "/home/shpark/prj-denovo/baselines"
CHUNK = 20

METHODS = [
    ("AR", "AR"),
    ("Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD", "DiffSBDD"),
    ("DecompDiff", "DecompDiff_ref_prior"),
    ("FuncBind", "FuncBind"),
]

SEL = json.load(open(f"{BASE}/_shared_data/density_pockets_selected.json"))


def load_meta(baseline):
    meta = torch.load(f"{BASE}/_meta/{baseline}.pt", weights_only=False)
    part2 = f"{BASE}/_meta/{baseline}_part2.pt"
    if os.path.exists(part2):
        other = torch.load(part2, weights_only=False)
        meta = [a + b for a, b in zip(meta, other)]
    return meta


def clean(x):
    """JSON has no NaN; a failed UFF relaxation is null, which is not the same as 0."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def export(label, baseline):
    meta = load_meta(baseline)
    pockets, mols = {}, []
    for path in sorted(glob.glob(f"{BASE}/_posecheck/{baseline}/pc_[0-9][0-9][0-9]_[0-9][0-9].pt")):
        stem = os.path.basename(path)[3:-3]
        idx, cid = (int(v) for v in stem.split("_"))
        d = torch.load(path, weights_only=False)
        entries = meta[idx][cid * CHUNK:(cid + 1) * CHUNK]
        if len(entries) != d["n"] or entries[0]["ligand_filename"] != d["ligand_filename"]:
            raise SystemExit(f"chunk/meta mismatch at {label} pocket {idx} chunk {cid}")
        pockets[str(idx)] = d["ligand_filename"]
        strain = np.asarray(d["strain"], dtype=float)
        clashes = np.asarray(d["clashes"], dtype=float)
        for k, e in enumerate(entries):
            mol = e.get("mol")
            if mol is None:
                continue
            mols.append({
                "p": idx,
                "n": int(mol.GetNumAtoms()),
                "s": clean(strain[k]) if k < strain.size else None,
                "c": clean(clashes[k]) if k < clashes.size else None,
            })

    doc = {
        "method": label,
        "source": f"prj-denovo/baselines/_posecheck/{baseline}",
        "protocol": "PoseCheck on the pose as generated; receptors protonated with pdb2pqr; "
                    "full *_rec.pdb receptor",
        "fields": {"p": "pocket index, 0-99, matches split_by_name.pt['test'] order",
                   "n": "heavy-atom count of the generated molecule",
                   "s": "UFF strain energy, kcal/mol; null = relaxation failed",
                   "c": "steric clashes with the receptor; null = not computed"},
        "n_pockets": len(pockets),
        "n_molecules": len(mols),
        "n_strain": sum(1 for m in mols if m["s"] is not None),
        "density79_pockets": SEL["indices"],
        "pocket_ligand_filename": pockets,
        "molecules": mols,
    }
    out = f"{OUT}/posecheck_{label}.json"
    json.dump(doc, open(out, "w"), separators=(",", ":"))
    size = os.path.getsize(out) / 1e6
    n79 = sum(1 for m in mols if m["p"] in set(SEL["indices"]))
    print(f"{label:12s} pockets {doc['n_pockets']:3d}  mols {doc['n_molecules']:5d} "
          f"(79-subset {n79:5d})  strain {doc['n_strain']:5d}  {size:5.1f} MB  -> {os.path.basename(out)}")


os.makedirs(OUT, exist_ok=True)
for label, baseline in METHODS:
    export(label, baseline)
