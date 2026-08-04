"""manifest.py — build the HBGSA sample manifest from PDBbind data.

Membership AND the train/val/test assignment come from a frozen split manifest
(voxbind/splits/<scheme>.csv : pid,split — default lp_edrscc_v2). Affinity labels
and per-complex metadata (smiles / seq / pK / CL1 / covalent) come from
LP_PDBBind.csv. Structure-file paths are resolved across the targetdiff
PDBbind-v2020 tree; complexes without protein+ligand on disk are dropped, and
missing SMILES are resolved from the ligand SDF via RDKit.

Output columns: pdb_id, new_split, pK, CL1, covalent, smiles, seq, + file paths.
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import DEFAULT_SPLIT_CSV, LP_CSV, complex_files


def _smiles_from_sdf(sdf_path, mol2_path=None) -> str | None:
    """Fallback SMILES via RDKit when LP_PDBBind.csv has none.

    Robust chain: strict SDF parse → relaxed SDF parse (best-effort sanitize,
    else canonicalize as-is) → mol2. The PDBbind ligand SDFs include many
    metal/valence cases RDKit rejects under strict sanitization; the relaxed
    path recovers them so the full split stays intact."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None

    def _smiles(mol):
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol) or None
        except Exception:
            return None

    # 1) strict SDF
    try:
        s = _smiles(next(iter(Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)), None))
        if s:
            return s
    except Exception:
        pass
    # 2) relaxed SDF (best-effort sanitize, else canonicalize the raw mol)
    try:
        mol = next(iter(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=True)), None)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            s = _smiles(mol)
            if s:
                return s
    except Exception:
        pass
    # 3) mol2
    if mol2_path is not None:
        try:
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
                s = _smiles(mol)
                if s:
                    return s
        except Exception:
            pass
    return None


def _to_bool(s: pd.Series) -> pd.Series:
    """Robustly coerce a CSV column ('True'/'False'/bool/NaN) to bool."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_manifest(
    *,
    split_csv=DEFAULT_SPLIT_CSV,    # frozen pid,split manifest (membership + split)
    cl1_only: bool = False,         # the split already defines the universe; CL1 is opt-in
    drop_covalent: bool = True,
    require_smiles: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the filtered manifest DataFrame keyed on the split-CSV's train/val/test."""
    # ── split manifest: defines who is in and their train/val/test bucket ──
    sp = pd.read_csv(split_csv)
    sp["pid"] = sp["pid"].astype(str).str.lower()
    split_map = dict(zip(sp["pid"], sp["split"]))
    keep_pids = set(sp["pid"])
    if verbose:
        print(f"  split manifest: {split_csv.name if hasattr(split_csv, 'name') else split_csv}  "
              f"({len(keep_pids)} pids)")

    # ── labels + metadata from LP_PDBBind.csv ──
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    lp["pdb_id"] = lp["pdb_id"].astype(str).str.lower()
    lp = lp[lp["pdb_id"].isin(keep_pids)].copy()
    df = lp[["pdb_id", "smiles", "seq", "pK", "CL1", "covalent"]].copy()
    df["CL1"] = _to_bool(df["CL1"])
    df["covalent"] = _to_bool(df["covalent"])
    df["new_split"] = df["pdb_id"].map(split_map)              # split CSV overrides

    def _drop(mask, reason):
        n = int((~mask).sum())
        if verbose and n:
            print(f"  drop {n:5d}  ({reason})")
        return df[mask]

    df = _drop(df["new_split"].isin(["train", "val", "test"]), "split not in {train,val,test}")
    df = _drop(df["pK"].notna(), "missing pK")
    if drop_covalent:
        df = _drop(~df["covalent"], "covalent")
    if cl1_only:
        df = _drop(df["CL1"], "not CL1-clean")

    # ── resolve structure file paths; require protein + ligand on disk ──
    paths = df["pdb_id"].map(complex_files)
    df = df.assign(
        protein_pdb=paths.map(lambda p: str(p["protein"])),
        pocket_pdb=paths.map(lambda p: str(p["pocket"])),
        ligand_sdf=paths.map(lambda p: str(p["ligand_sdf"])),
        ligand_mol2=paths.map(lambda p: str(p["ligand_mol2"])),
    )
    exists = paths.map(lambda p: p["protein"].exists() and p["ligand_sdf"].exists())
    df = _drop(exists, "protein/ligand file missing")

    # ── SMILES fallback from SDF for the rows LP_PDBBind.csv leaves null ──
    null_smi = df["smiles"].isna()
    if null_smi.any():
        if verbose:
            print(f"  resolving {int(null_smi.sum())} missing SMILES from ligand SDF (RDKit)…")
        df.loc[null_smi, "smiles"] = df.loc[null_smi].apply(
            lambda r: _smiles_from_sdf(r["ligand_sdf"], r["ligand_mol2"]), axis=1)
    if require_smiles:
        df = _drop(df["smiles"].notna(), "SMILES unresolved")

    df = df[["pdb_id", "new_split", "pK", "CL1", "covalent",
             "smiles", "seq", "protein_pdb", "pocket_pdb", "ligand_sdf", "ligand_mol2"]].reset_index(drop=True)
    return df


def _print_split_table(df: pd.DataFrame) -> None:
    g = df.groupby("new_split").size()
    print("\n  split sizes:")
    for s in ("train", "val", "test"):
        print(f"    {s:5s} {int(g.get(s, 0)):5d}")
    print(f"    {'TOTAL':5s} {len(df):5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build + summarize the HBGSA PDBbind manifest")
    ap.add_argument("--split_csv", default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--cl1_only", action="store_true", help="restrict to LP_PDBBind CL1-clean subset")
    ap.add_argument("--keep_covalent", action="store_true", help="do NOT drop covalent complexes")
    args = ap.parse_args()

    from pathlib import Path
    print(f"=== HBGSA manifest (split={Path(args.split_csv).name}, cl1_only={args.cl1_only}) ===")
    m = build_manifest(split_csv=Path(args.split_csv), cl1_only=args.cl1_only,
                       drop_covalent=not args.keep_covalent)
    _print_split_table(m)
    print("\n  sample row:")
    r = m.iloc[0]
    print(f"    {r.pdb_id}  split={r.new_split}  pK={r.pK}  smiles={str(r.smiles)[:48]}…  seq_len={len(str(r.seq))}")
