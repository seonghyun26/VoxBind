#!/usr/bin/env python3
"""Stage the official LP-PDBBind BDB2020+ benchmark for VoxBind preprocessing.

The LP-PDBBind archive contains full proteins and ligand SDF files. VoxBind's
PDBbind voxelizer expects a lower-case directory per PDB ID with a ligand SDF
and a pre-cut pocket PDB, so this script creates that deterministic adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "data" / "bdb2020plus"
DEFAULT_SOURCE = DEFAULT_ROOT / "raw" / "BDB2020+"
OFFICIAL_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/THGLab/LP-PDBBind/"
    "master/dataset/BDB2020%2B.tgz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ligand_coordinates(sdf_path: Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None or not mol.GetNumConformers():
        raise ValueError(f"cannot read a 3D ligand from {sdf_path}")
    conformer = mol.GetConformer()
    coords = np.asarray(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    heavy = np.asarray(
        [mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in range(mol.GetNumAtoms())]
    )
    if heavy.any():
        coords = coords[heavy]
    if not np.isfinite(coords).all():
        raise ValueError(f"non-finite ligand coordinates in {sdf_path}")
    return coords


def pdb_atom_record(line: str):
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
        return None
    altloc = line[16:17]
    if altloc not in (" ", "A"):
        return None
    try:
        xyz = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        )
    except ValueError:
        return None
    element = line[76:78].strip() if len(line) >= 78 else ""
    if not element:
        element = "".join(ch for ch in line[12:16] if ch.isalpha())[:1]
    residue = (line[21:22], line[22:26], line[26:27], line[17:20])
    return xyz, element.upper(), residue


def write_residue_pocket(
    protein_path: Path,
    ligand_xyz: np.ndarray,
    output_path: Path,
    cutoff: float,
) -> int:
    lines = protein_path.read_text(errors="replace").splitlines(keepends=True)
    parsed = [pdb_atom_record(line) for line in lines]
    heavy_records = [record for record in parsed if record and record[1] not in ("H", "D")]
    if not heavy_records:
        raise ValueError(f"no protein heavy atoms in {protein_path}")

    protein_xyz = np.stack([record[0] for record in heavy_records])
    close = np.any(
        np.sum((protein_xyz[:, None, :] - ligand_xyz[None, :, :]) ** 2, axis=2)
        <= cutoff * cutoff,
        axis=1,
    )
    residues = {
        record[2] for record, is_close in zip(heavy_records, close) if is_close
    }
    if not residues:
        raise ValueError(f"no protein residues within {cutoff:g} A in {protein_path}")

    pocket_lines = [
        line for line, record in zip(lines, parsed) if record and record[2] in residues
    ]
    output_path.write_text("".join(pocket_lines) + "END\n")
    return len(pocket_lines)


def stage(source_dir: Path, output_root: Path, cutoff: float) -> None:
    labels_path = source_dir / "BDB2020+.csv"
    structures_dir = source_dir / "dataset"
    archive_path = source_dir.parent / "BDB2020+.tgz"
    if not labels_path.is_file() or not structures_dir.is_dir():
        raise FileNotFoundError(
            f"expected BDB2020+.csv and dataset/ below {source_dir}"
        )

    upstream_labels = pd.read_csv(labels_path)
    required = {"pdbid", "value", "accurate", "pKa"}
    missing = required.difference(upstream_labels.columns)
    if missing:
        raise ValueError(f"missing official label columns: {sorted(missing)}")

    # The paper's benchmark is the accurate-affinity subset.  The currently
    # released CSV already contains exactly those 115 rows, but filter
    # explicitly so a future archive containing the wider candidate pool can
    # never silently change the evaluated cohort.
    accurate = (
        upstream_labels["accurate"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    labels = upstream_labels[accurate].copy().reset_index(drop=True)
    if labels.empty:
        raise ValueError("BDB2020+ contains no rows with accurate=True")
    if labels["pdbid"].str.lower().duplicated().any():
        raise ValueError("BDB2020+ contains duplicate PDB IDs")

    staged_dir = output_root / "structures"
    staged_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for record in labels.to_dict("records"):
        source_id = str(record["pdbid"]).upper()
        pid = source_id.lower()
        source_complex = structures_dir / source_id
        ligand_path = source_complex / "ligand.sdf"
        protein_path = source_complex / "protein.pdb"
        destination = staged_dir / pid
        destination.mkdir(parents=True, exist_ok=True)
        try:
            ligand_xyz = ligand_coordinates(ligand_path)
            shutil.copy2(ligand_path, destination / f"{pid}_ligand.sdf")
            n_pocket_atoms = write_residue_pocket(
                protein_path,
                ligand_xyz,
                destination / f"{pid}_pocket.pdb",
                cutoff,
            )
            rows.append(
                {
                    "pdb_id": pid,
                    "has_struct": True,
                    "pK": float(record["pKa"]),
                    "benchmark": "BDB2020+",
                    "source_pdb_id": source_id,
                    "value": float(record["value"]),
                    "accurate": True,
                    "n_pocket_atoms": n_pocket_atoms,
                    "note": "",
                }
            )
        except Exception as exc:
            failures.append((pid, str(exc)))
            rows.append(
                {
                    "pdb_id": pid,
                    "has_struct": False,
                    "pK": float(record["pKa"]),
                    "benchmark": "BDB2020+",
                    "source_pdb_id": source_id,
                    "value": float(record["value"]),
                    "accurate": True,
                    "n_pocket_atoms": 0,
                    "note": str(exc),
                }
            )

    index_path = output_root / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "benchmark": "BDB2020+",
        "official_archive_url": OFFICIAL_ARCHIVE_URL,
        "source_csv": str(labels_path.resolve()),
        "source_archive": str(archive_path.resolve()) if archive_path.exists() else None,
        "source_archive_sha256": sha256(archive_path) if archive_path.exists() else None,
        "pocket_cutoff_angstrom": cutoff,
        "n_upstream_rows": len(upstream_labels),
        "n_official": len(labels),
        "n_staged": len(rows) - len(failures),
        "n_failed": len(failures),
    }
    (output_root / "staging_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    if failures:
        print("Failures:")
        for pid, message in failures:
            print(f"  {pid}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pocket_cutoff", type=float, default=10.0)
    args = parser.parse_args()
    stage(args.source_dir, args.output_root, args.pocket_cutoff)


if __name__ == "__main__":
    main()
