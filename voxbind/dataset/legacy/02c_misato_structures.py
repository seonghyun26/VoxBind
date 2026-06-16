"""02c_misato_structures.py — synthesize PDBbind-style structure files for
MISATO complexes we lack processed structures for, so the existing 01b
voxelizer can consume them unchanged.

Two artifacts per complex (mirrors structures/pbpp-2020/{pid}/):
    {pid}_ligand.sdf   V2000  — ligand atoms/coords/bonds straight from QM.hdf5
    {pid}_pocket.pdb   PDB    — protein ATOM records within CUTOFF of the ligand,
                                sliced from the deposited PDB

Why this is sound (validated 260612):
  * QM.hdf5 ligand coordinates are *identical* to the PDBbind crystal-frame
    ligand (0.000 A centroid offset over 400 overlap complexes) — so the QM
    ligand, the deposited PDB, and the PDBe EDS map all share one frame. No
    Kabsch, no ligand-identity guessing (QM gives the exact ligand atoms).
  * The voxelizer only uses C/O/N/S for the pocket (no H channel), so the
    deposited PDB lacking hydrogens is irrelevant.

Envs: this script needs h5py (cellmaes / compuworks env), NOT the voxbind env.
It only writes text files; voxelization runs separately under voxbind.

Usage
-----
    PY=/home/shpark/.conda/envs/cellmaes/bin/python
    # validate: build into a temp dir + report (run voxel compare separately)
    $PY dataset/02c_misato_structures.py validate --pids 10gs 1a30 ... --out /tmp/valbuild
    # build: for an id list, write structures + a has_struct=True index CSV
    $PY dataset/02c_misato_structures.py build --ids /tmp/freetier/structless_all.txt \
        --out_struct dataset/data/pdbbind/structures/misato_qm_built \
        --out_index  dataset/data/pdbbind/misato_built_index.csv
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent.parent
QM_PATH = HERE / "data" / "misato" / "QM.hdf5"
PDB_DIR = HERE / "data" / "pdb"                       # deposited PDBs (pdbe entry-files)
POCKET_CUTOFF = 14.0   # A, residue-complete; >= voxel-box half-diagonal (8√3≈13.9) so
                       # the 16 A crop is fully covered (matches PDBbind, whose atom
                       # positions are identical — only inclusion differs)

# atomic number -> element symbol (covers ligand + pocket + common metals/ions)
Z2SYM = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 33: "As", 34: "Se", 35: "Br", 42: "Mo", 48: "Cd", 53: "I",
    78: "Pt", 80: "Hg",
}


def qm_ligand_arrays(g):
    """(coords (N,3) float, Z (N,) int, bonds (M,2) int 1-indexed deduped)."""
    Z = np.array([int(x) for x in g["atom_properties/atom_names"][:]])
    xyz = g["atom_properties/atom_properties_values"][:, :3].astype(float)
    b = g["atom_properties/bonds"][:]                 # (K,3): i, j, order (0-indexed, both dirs)
    seen, bonds = set(), []
    for i, j, order in b:
        i, j, order = int(i), int(j), int(round(order)) or 1
        a, c = (i, j) if i < j else (j, i)
        if a == c or (a, c) in seen:
            continue
        seen.add((a, c))
        bonds.append((a + 1, c + 1, order))           # SDF is 1-indexed
    return xyz, Z, bonds


def write_ligand_sdf(pid, g, path):
    xyz, Z, bonds = qm_ligand_arrays(g)
    n_atoms, n_bonds = len(Z), len(bonds)
    lines = [f"{pid}_ligand", "  -misato-qm- ", ""]
    lines.append(f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000")
    for (x, y, z), zi in zip(xyz, Z):
        sym = Z2SYM.get(zi, "C")
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {sym:<3s} 0  0  0  0  0")
    for a, c, order in bonds:
        lines.append(f"{a:>3d}{c:>3d}{order:>3d}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    path.write_text("\n".join(lines) + "\n")
    return n_atoms


def read_deposited_protein_atoms(pdb_path):
    """Protein ATOM records → (coords (N,3), raw_lines). Heavy + H both kept;
    only first altloc (' ' or 'A'). HETATM (ligand/water/ion) excluded."""
    coords, raws = [], []
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM  "):
                continue
            alt = ln[16]
            if alt not in (" ", "A"):
                continue
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            coords.append((x, y, z)); raws.append(ln.rstrip("\n"))
    return np.asarray(coords, float), raws


def write_pocket_pdb(pid, lig_xyz, pdb_path, out_path, cutoff=POCKET_CUTOFF):
    """Residue-complete protein pocket within `cutoff` of any ligand atom."""
    coords, raws = read_deposited_protein_atoms(pdb_path)
    if len(coords) == 0:
        return 0, 0
    # min distance of each protein atom to the ligand
    # (chunk to bound memory on large proteins)
    near = np.zeros(len(coords), bool)
    for s in range(0, len(coords), 4096):
        chunk = coords[s:s + 4096]
        d = np.sqrt(((chunk[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1)).min(1)
        near[s:s + 4096] = d <= cutoff
    # residue-complete: keep all atoms of any residue with a near atom
    # residue key = chain(21) + resseq(22:26) + icode(26)
    res_keys = [r[21] + r[22:27] for r in raws]
    keep_res = {res_keys[i] for i in np.where(near)[0]}
    sel = [r for r, k in zip(raws, res_keys) if k in keep_res]
    header = [f"HEADER    {pid.upper()}_POCKET", f"COMPND    {pid.upper()}_POCKET",
              "REMARK    GENERATED BY 02c_misato_structures.py (QM-ligand pocket slice)"]
    out_path.write_text("\n".join(header + sel) + "\nEND\n")
    return len(sel), len(keep_res)


def build_one(pid, qm, out_struct, pdb_dir):
    g = qm.get(pid.upper())
    if g is None:
        return pid, "no_qm"
    pdb_path = pdb_dir / f"{pid}.pdb"
    if not pdb_path.exists():
        return pid, "no_pdb"
    cdir = out_struct / pid
    cdir.mkdir(parents=True, exist_ok=True)
    na = write_ligand_sdf(pid, g, cdir / f"{pid}_ligand.sdf")
    xyz, _, _ = qm_ligand_arrays(g)
    npoc, nres = write_pocket_pdb(pid, xyz, pdb_path, cdir / f"{pid}_pocket.pdb")
    if npoc == 0:
        return pid, "empty_pocket"
    return pid, f"ok lig={na} poc={npoc} res={nres}"


def cmd_build(args):
    ids = [l.strip().lower() for l in open(args.ids) if l.strip()]
    out_struct = Path(args.out_struct)
    pdb_dir = Path(args.pdb_dir)
    qm = h5py.File(QM_PATH, "r")
    ok, reasons = [], {}
    for n, pid in enumerate(ids, 1):
        pid_, msg = build_one(pid, qm, out_struct, pdb_dir)
        cat = msg.split()[0]
        reasons[cat] = reasons.get(cat, 0) + 1
        if cat == "ok":
            ok.append(pid_)
        if n % 500 == 0:
            print(f"  {n}/{len(ids)}  {reasons}", flush=True)
    qm.close()
    # write a has_struct=True index for the built complexes (01b reads pdb_id+has_struct)
    import csv
    with open(args.out_index, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pdb_id", "has_struct"])
        for pid in ok:
            w.writerow([pid, True])
    print(f"built {len(ok)}/{len(ids)}  reasons={reasons}")
    print(f"struct dir: {out_struct}")
    print(f"index csv : {args.out_index}  ({len(ok)} rows)")


def cmd_validate(args):
    out = Path(args.out)
    qm = h5py.File(QM_PATH, "r")
    pdb_dir = Path(args.pdb_dir)
    for pid in args.pids:
        print(build_one(pid, qm, out, pdb_dir))
    qm.close()
    print(f"\nwrote to {out} — voxel-compare against pbpp-2020 separately (voxbind env)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build")
    pb.add_argument("--ids", required=True)
    pb.add_argument("--out_struct", required=True)
    pb.add_argument("--out_index", required=True)
    pb.add_argument("--pdb_dir", default=str(PDB_DIR))
    pb.set_defaults(func=cmd_build)
    pv = sub.add_parser("validate")
    pv.add_argument("--pids", nargs="+", required=True)
    pv.add_argument("--out", default="/tmp/valbuild")
    pv.add_argument("--pdb_dir", default=str(PDB_DIR))
    pv.set_defaults(func=cmd_validate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
