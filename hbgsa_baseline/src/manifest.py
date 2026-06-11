"""manifest.py — build the HBGSA sample manifest from PDBbind data only.

Joins index.csv (has_struct / new_split / pK / CL1 / covalent) with
LP_PDBBind.csv (smiles / seq) on pdb_id, keeps complexes that have structure
files on disk, and resolves missing SMILES from the ligand SDF via RDKit.

Output columns: pdb_id, new_split, pK, CL1, covalent, smiles, seq, + file paths.
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import INDEX_CSV, LP_CSV, complex_files


def _smiles_from_sdf(sdf_path) -> str | None:
    """Fallback SMILES via RDKit when LP_PDBBind.csv has none."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
        mol = next(iter(suppl), None)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def build_manifest(
    *,
    cl1_only: bool = True,        # CL1-clean filter is the project default
    drop_covalent: bool = True,
    require_smiles: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the filtered manifest DataFrame keyed on LP-PDBbind new_split."""
    idx = pd.read_csv(INDEX_CSV)
    idx["pdb_id"] = idx["pdb_id"].str.lower()

    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pdb_id"})
    lp["pdb_id"] = lp["pdb_id"].str.lower()
    lp = lp[["pdb_id", "smiles", "seq"]]

    df = idx.merge(lp, on="pdb_id", how="left")

    def _drop(mask, reason):
        n = int((~mask).sum())
        if verbose and n:
            print(f"  drop {n:5d}  ({reason})")
        return df[mask]

    df = _drop(df["has_struct"] == True, "no structure on disk")            # noqa: E712
    df = _drop(df["new_split"].isin(["train", "val", "test"]), "split not in {train,val,test}")
    df = _drop(df["pK"].notna(), "missing pK")
    if drop_covalent:
        df = _drop(df["covalent"] != True, "covalent")                      # noqa: E712
    if cl1_only:
        df = _drop(df["CL1"] == True, "not CL1-clean")                      # noqa: E712

    # resolve file paths; verify protein + ligand exist
    paths = df["pdb_id"].map(complex_files)
    df = df.assign(
        protein_pdb=paths.map(lambda p: str(p["protein"])),
        ligand_sdf=paths.map(lambda p: str(p["ligand_sdf"])),
        ligand_mol2=paths.map(lambda p: str(p["ligand_mol2"])),
    )
    exists = paths.map(lambda p: p["protein"].exists() and p["ligand_sdf"].exists())
    df = _drop(exists, "protein/ligand file missing")

    # SMILES fallback from SDF for the rows LP_PDBBind.csv leaves null
    null_smi = df["smiles"].isna()
    if null_smi.any():
        if verbose:
            print(f"  resolving {int(null_smi.sum())} missing SMILES from ligand SDF (RDKit)…")
        df.loc[null_smi, "smiles"] = df.loc[null_smi, "ligand_sdf"].map(_smiles_from_sdf)
    if require_smiles:
        df = _drop(df["smiles"].notna(), "SMILES unresolved")

    df = df[["pdb_id", "new_split", "pK", "CL1", "covalent",
             "smiles", "seq", "protein_pdb", "ligand_sdf", "ligand_mol2"]].reset_index(drop=True)
    return df


def _print_split_table(df: pd.DataFrame) -> None:
    g = df.groupby("new_split").size()
    print("\n  split sizes:")
    for s in ("train", "val", "test"):
        print(f"    {s:5s} {int(g.get(s, 0)):5d}")
    print(f"    {'TOTAL':5s} {len(df):5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build + summarize the HBGSA PDBbind manifest")
    ap.add_argument("--no_cl1", action="store_true", help="disable the default CL1-clean filter")
    ap.add_argument("--keep_covalent", action="store_true", help="do NOT drop covalent complexes")
    args = ap.parse_args()

    print("=== HBGSA manifest (PDBbind new_split, CL1-default) ===")
    m = build_manifest(cl1_only=not args.no_cl1, drop_covalent=not args.keep_covalent)
    _print_split_table(m)
    print("\n  sample row:")
    r = m.iloc[0]
    print(f"    {r.pdb_id}  split={r.new_split}  pK={r.pK}  smiles={r.smiles[:48]}…  seq_len={len(r.seq)}")
