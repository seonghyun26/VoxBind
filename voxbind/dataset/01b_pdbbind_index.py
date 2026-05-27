"""01b_pdbbind_index.py — Phase 2: extract PDBbind v2020 refined set + build index.

Reads raw downloads placed by 01a in dataset/data/pdbbind/raw/:
  - pbpp-2020.zip       (HF photonmz/pdbbindpp-2020, ~5,316 refined complexes)
  - LP_PDBBind.csv      (THGLab/LP-PDBBind metadata + leak-proof splits)

Extracts the zip into dataset/data/pdbbind/structures/<pdb_id>/{protein,pocket,ligand}.*
and writes a single canonical index file:
  - dataset/data/pdbbind/index.csv
    columns: pdb_id, set, pK, new_split, CL1, CL2, CL3, covalent, has_struct

Usage
-----
    cd voxbind
    python dataset/01b_pdbbind_index.py
    python dataset/01b_pdbbind_index.py --force_extract   # re-extract zip

Outputs
-------
    dataset/data/pdbbind/structures/<pdb_id>/...   one dir per refined complex
    dataset/data/pdbbind/index.csv
    dataset/data/pdbbind/missing.txt               (pdb_ids in CSV but no struct)
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
RAW_DIR     = PDBBIND_DIR / "raw"
STRUCT_DIR  = PDBBIND_DIR / "structures"
INDEX_OUT   = PDBBIND_DIR / "index.csv"

ZIP_NAME = "pbpp-2020.zip"
CSV_NAME = "LP_PDBBind.csv"

# Per-complex file naming pattern (substituted with pdb_id at check time)
REQUIRED_PER_PDBID = [
    "{pdb_id}_protein.pdb",
    "{pdb_id}_pocket.pdb",
    "{pdb_id}_ligand.mol2",
    "{pdb_id}_ligand.sdf",
]


# ── Extract ────────────────────────────────────────────────────────────────────

def is_populated(d: Path) -> bool:
    if not d.exists():
        return False
    return any(d.iterdir())


def extract_zip(zip_path: Path, dst_root: Path, force: bool) -> None:
    """Extract pbpp-2020.zip into dst_root/<pdb_id>/... .

    The zip's internal layout is determined at extract time — we just unpack
    everything and then normalize per-complex paths if needed.
    """
    if is_populated(dst_root) and not force:
        n_dirs = sum(1 for p in dst_root.iterdir() if p.is_dir())
        print(f"  [skip] {dst_root}/  ({n_dirs} entries, --force_extract to redo)")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {zip_path.name}  →  {dst_root}/")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=zip_path.name, unit="file"):
            zf.extract(m, dst_root)


def detect_struct_root(struct_root: Path) -> Path:
    """Find the directory that actually holds <pdb_id>/ children.

    The zip may extract to struct_root/ directly, or into one of several
    sibling subdirs (e.g. pbpp-2020/ plus readme/). Pick whichever path has
    the most 4-char alnum children.
    """
    if not struct_root.exists():
        return struct_root

    def n_pdbids(d: Path) -> int:
        try:
            return sum(1 for p in d.iterdir()
                       if p.is_dir() and len(p.name) == 4 and p.name.isalnum())
        except OSError:
            return 0

    best = struct_root
    best_count = n_pdbids(struct_root)
    for sub in struct_root.iterdir():
        if sub.is_dir():
            c = n_pdbids(sub)
            if c > best_count:
                best = sub
                best_count = c
    return best


# ── Index assembly ─────────────────────────────────────────────────────────────

def load_lp_pdbbind(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # First column is the pdb_id (saved without header → "Unnamed: 0").
    # Also accept "pdbid" if a newer version of the CSV uses that.
    pdb_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else "pdbid"
    df = df.rename(columns={pdb_col: "pdb_id"})

    # pK column — LP-PDBbind uses "value"; tolerate variants from other releases.
    pK_col = next((c for c in ["value", "-logKd/Ki", "pK", "pK_value"]
                   if c in df.columns), None)
    if pK_col is None:
        raise KeyError(f"Could not locate pK column in {csv_path}; "
                       f"saw columns: {list(df.columns)}")
    df = df.rename(columns={pK_col: "pK"})

    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def attach_struct_membership(df: pd.DataFrame, struct_root: Path) -> pd.DataFrame:
    if not struct_root.exists():
        df["has_struct"] = False
        return df

    def has(pid: str) -> bool:
        d = struct_root / pid
        if not d.exists():
            return False
        return all((d / fn.format(pdb_id=pid)).exists() for fn in REQUIRED_PER_PDBID)

    df["has_struct"] = df["pdb_id"].apply(has)
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract PDBbind v2020 refined set and build index.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_dir",    default=str(RAW_DIR))
    p.add_argument("--struct_dir", default=str(STRUCT_DIR))
    p.add_argument("--index_out",  default=str(INDEX_OUT))
    p.add_argument("--force_extract", action="store_true",
                   help="Re-extract pbpp-2020.zip even if structures/ is populated")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir    = Path(args.raw_dir).resolve()
    struct_dir = Path(args.struct_dir).resolve()
    index_out  = Path(args.index_out).resolve()

    print("=== PDBbind v2020 index build ===")
    print(f"  raw_dir    : {raw_dir}")
    print(f"  struct_dir : {struct_dir}")
    print(f"  index_out  : {index_out}")

    # Sanity: raw downloads present?
    zip_path = raw_dir / ZIP_NAME
    csv_path = raw_dir / CSV_NAME
    missing = [p for p in (zip_path, csv_path) if not p.exists()]
    if missing:
        print("\n[error] missing input(s):")
        for p in missing:
            print(f"    - {p}")
        print("Run dataset/01a_pdbbind_acquire.py first.")
        sys.exit(1)

    # Extract
    print("\n── Extract ─────────────────────────────────────────────────────")
    extract_zip(zip_path, struct_dir, force=args.force_extract)

    # Locate the actual <pdb_id>/ root inside what we extracted
    real_root = detect_struct_root(struct_dir)
    if real_root != struct_dir:
        print(f"  [info] real structure root is {real_root.relative_to(struct_dir.parent)}")

    # Load LP-PDBbind metadata
    print("\n── Load LP_PDBBind.csv ─────────────────────────────────────────")
    df = load_lp_pdbbind(csv_path)
    print(f"  {len(df):,} rows; columns: {list(df.columns)}")

    # Attach structural membership
    df = attach_struct_membership(df, real_root)

    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  total rows         : {len(df):,}")
    print(f"  has_struct (True)  : {df['has_struct'].sum():,}")
    print(f"  has_struct (False) : {(~df['has_struct']).sum():,}")
    if "category" in df.columns:
        print("\n  category  ×  has_struct:")
        print(df.groupby(["category", "has_struct"]).size().unstack(fill_value=0).to_string())
    if "new_split" in df.columns:
        print("\n  new_split  ×  has_struct:")
        print(df.groupby(["new_split", "has_struct"], dropna=False).size().unstack(fill_value=0).to_string())

    # Write missing list (entries in CSV but no structure on disk; mostly general-set)
    miss_path = PDBBIND_DIR / "missing.txt"
    miss_ids = df.loc[~df["has_struct"], "pdb_id"].tolist()
    if miss_ids:
        miss_path.write_text("\n".join(miss_ids) + "\n")
        print(f"\n  {len(miss_ids):,} pdb_ids not on disk  →  {miss_path}")

    # Write final index — keep useful columns, in stable order
    keep_cols = [c for c in
                 ["pdb_id", "has_struct", "category", "pK", "new_split",
                  "CL1", "CL2", "CL3", "covalent",
                  "resolution", "date", "type", "kd/ki"]
                 if c in df.columns]
    df_out = df[keep_cols].copy()
    df_out.to_csv(index_out, index=False)
    print(f"\n[write] {index_out}  ({len(df_out):,} rows, {len(keep_cols)} cols)")

    print("\nNext (Phase 3): dataset/01c_pdbbind_density_download.py")


if __name__ == "__main__":
    main()
