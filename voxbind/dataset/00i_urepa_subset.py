"""00i_urepa_subset.py — build the U-REPA alignment subset.

The alignment loss (L_align) is applied ONLY to CrossDocked samples with matching
experimental density: native (tt_min) poses whose receptor 2Fo-Fc map is on disk.
The denoise loss stays on the full CrossDocked set — this subset only gates L_align.

Filter chain (plan §4 "Data split of the two losses" + "Split hygiene"):
    train samples
      → native (tt_min: ligand source PDB == receptor PDB, energy-min crystal pose)
      → receptor 2Fo-Fc CCP4 map present on disk
      → receptor/ligand PDB does NOT overlap any TEST pocket  (explicit leak guard)

Writes  dataset/data/pretrain/urepa_subset.pt :
    { "pids":[...], "receptor_pdb":[...], "train_index":[...],
      "test_excluded_pdbs":[...], "counts":{...} }
keyed by the sample's pocket id (pocket10 path) — the stable cross-server key the
token cache (00j) and the training loop match on.

Usage:
    python dataset/00i_urepa_subset.py            # build + write
    python dataset/00i_urepa_subset.py --dry_run  # counts only, no write
"""
import argparse
import re
from pathlib import Path

import torch

VOXDATA = Path(__file__).resolve().parent / "data"
CCP4_DIR = VOXDATA / "ccp4"
OUT = VOXDATA / "pretrain" / "urepa_subset.pt"

# CrossDocked provenance: {receptor}_{chain}_rec_{ligand-PDB}_{resname}_lig_[itN_]*tt_{min|docked}_{rank}
_PROV_RE = re.compile(
    r"([0-9a-zA-Z]{4})_\w+?_rec_([0-9a-zA-Z]{4})_\w+?_lig_(?:it\d+_)*tt_(min|docked)")


def parse_provenance(pocket_id):
    """(receptor_pdb, ligand_src_pdb, pose) lowercased, or None if unparsable."""
    m = _PROV_RE.match(pocket_id.split("/")[-1])
    return (m.group(1).lower(), m.group(2).lower(), m.group(3)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(VOXDATA))
    ap.add_argument("--ccp4_dir", default=str(CCP4_DIR))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ccp4 = Path(args.ccp4_dir)
    train = torch.load(data_dir / "data_train.pt", weights_only=False)
    test = torch.load(data_dir / "data_test.pt", weights_only=False)

    # test-pocket PDBs to exclude (both receptor and ligand-source sides).
    test_pdbs = set()
    for p, _l in test:
        pr = parse_provenance(p["id"])
        if pr:
            test_pdbs.update({pr[0], pr[1]})

    have_map = {f.stem.lower()[:4] for f in ccp4.glob("*.ccp4")}

    pids, rec_pdbs, tr_idx = [], [], []
    n_native = n_native_map = n_excluded = 0
    for i, (p, _l) in enumerate(train):
        pr = parse_provenance(p["id"])
        if not (pr and pr[0] == pr[1] and pr[2] == "min"):     # native tt_min only
            continue
        n_native += 1
        rec = pr[0]
        if rec not in have_map:                                # density-available
            continue
        n_native_map += 1
        if rec in test_pdbs or pr[1] in test_pdbs:             # test-leak guard
            n_excluded += 1
            continue
        pids.append(p["id"]); rec_pdbs.append(rec); tr_idx.append(i)

    counts = dict(
        train_total=len(train), native_ttmin=n_native,
        native_with_map=n_native_map, test_excluded=n_excluded,
        subset=len(pids), test_pockets=len(test_pdbs),
    )
    print("=== U-REPA density subset ===")
    for k, v in counts.items():
        print(f"  {k:16s}: {v:,}")

    if args.dry_run:
        print("[dry_run] not writing"); return
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pids=pids, receptor_pdb=rec_pdbs, train_index=tr_idx,
                    test_excluded_pdbs=sorted(test_pdbs), counts=counts), out)
    print(f"→ wrote {len(pids):,} samples to {out}")


if __name__ == "__main__":
    main()
