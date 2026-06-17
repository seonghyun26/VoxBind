"""verify_split.py — Reproducibility guard for the canonical PDBbind affinity split.

The train/val/test partition is **100% the published LP-PDBBind `new_split` column**
(no seed, no resampling). On top of it the probe layers a stack of deterministic
membership filters; this script regenerates the resulting split from primary
artifacts and asserts the canonical counts + pid-set hashes, so a future
re-voxelization or EDS re-check that silently shifts the split fails loudly.

Canonical split (combined atoms+density models, `01c_pdbbind_probe.py`):

    train / val / test  =  2172 / 480 / 839

Constraint stack (all deterministic; mirrors build_dataset + the probe pool):
    1. new_split ∈ {train, val, test}                     (LP_PDBBind.csv)
    2. drop covalent == True                              (LP_PDBBind.csv)
    3. pK (`value`) not null                              (LP_PDBBind.csv)
    4. has_atoms  — atom crop built (element-supported)   (voxels/availability.csv)
    5. has_density — density crop built                   (voxels/availability.csv)
       (coords-only models use has_atoms only → 2414/514/922)

The two availability flags fold in the structure source (pbpp-2020 refined set),
the PDBe EDS availability (eds_cache.json), the element-support vocab
(pocket {C,O,N,S}, ligand {C,O,N,S,F,Cl,P} → metal pockets / B,Se ligands dropped),
and crop success. `voxels/availability.csv` is the single authoritative artifact;
the probe reads exactly `avail[has_atoms & has_density]`.

    cd voxbind
    python dataset/verify_split.py                # assert canonical; exit 1 on drift
    python dataset/verify_split.py --no-hash      # counts only (skip pid-set hash)
    python dataset/verify_split.py --show-dropped # list element-filter exclusions

Exit code 0 = reproduces the pinned split exactly; 1 = drift detected.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
LP_CSV      = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"
AVAIL_CSV   = PDBBIND_DIR / "voxels" / "availability.csv"

# ── Pinned canonical split (combined atoms+density; cl1_only=False) ──────────────
# Counts AND a sha256 over the sorted lowercase pid list per split. The hash
# catches "same count, different pids" drift that a count check alone would miss.
EXPECTED = {
    "train": (2172, "9844b3058c66fa14c40474aa0691b2080f7ddfe202f35f3d4747cd15444097fc"),
    "val":   (480,  "757ba66f0da97716b1e7583c2000d406d24f2bc95a6b10b5045073cf21db2bba"),
    "test":  (839,  "0d12ee18bd24c256d792fdf7525a70eb728b47739b1618a4ca0c2fac7647175a"),
}
# Coords-only reference pool (has_atoms, no density requirement); reported, not asserted.
COORDS_ONLY_EXPECTED = {"train": 2414, "val": 514, "test": 922}


def load_lp(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def base_frame(lp: pd.DataFrame) -> pd.DataFrame:
    """Rows surviving the split-independent filters (covalent / pK / new_split)."""
    df = lp[~lp["covalent"].astype(bool)].dropna(subset=["pK"])
    return df[df["new_split"].isin(["train", "val", "test"])]


def split_counts(df: pd.DataFrame, pool: set[str]) -> dict[str, list[str]]:
    sub = df[df["pdb_id"].isin(pool)]
    return {s: sorted(sub.loc[sub["new_split"] == s, "pdb_id"]) for s in ("train", "val", "test")}


def sha(pids: list[str]) -> str:
    return hashlib.sha256("\n".join(pids).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Assert the canonical PDBbind affinity split is reproducible.")
    ap.add_argument("--lp_csv",    default=str(LP_CSV))
    ap.add_argument("--avail_csv", default=str(AVAIL_CSV),
                    help="voxels/availability.csv (has_atoms/has_density), as the probe reads it")
    ap.add_argument("--no-hash", action="store_true", help="check counts only, skip pid-set hashes")
    ap.add_argument("--show-dropped", action="store_true",
                    help="list complexes dropped by the element-support filter")
    args = ap.parse_args()

    lp_csv, avail_csv = Path(args.lp_csv), Path(args.avail_csv)
    for p in (lp_csv, avail_csv):
        if not p.exists():
            print(f"[error] missing input: {p}")
            return 1

    lp    = load_lp(lp_csv)
    avail = pd.read_csv(avail_csv); avail["pdb_id"] = avail["pdb_id"].str.lower()
    base  = base_frame(lp)

    has_atoms   = set(avail.loc[avail["has_atoms"].astype(bool), "pdb_id"])
    has_density = set(avail.loc[avail["has_atoms"].astype(bool) & avail["has_density"].astype(bool), "pdb_id"])

    # ── Provenance: the deterministic constraint stack, stage by stage ──────────
    def line(label, d):
        print(f"  {label:46s} train={len(d['train']):5d} val={len(d['val']):5d} test={len(d['test']):5d}")

    print("=== Canonical PDBbind affinity split — reproducibility check ===")
    print(f"  LP_PDBBind : {lp_csv}")
    print(f"  avail      : {avail_csv}\n")
    print("  constraint stack (deterministic; partition = LP new_split column):")
    full = {s: sorted(lp.loc[lp["new_split"] == s, "pdb_id"]) for s in ("train", "val", "test")}
    line("1-3. new_split + drop covalent + pK",
         {s: sorted(base.loc[base["new_split"] == s, "pdb_id"]) for s in ("train", "val", "test")})
    coords = split_counts(base, has_atoms)
    line("  + has_atoms          (coords-only pool)", coords)
    canon  = split_counts(base, has_density)
    line("  + has_density        (CANONICAL combined)", canon)
    print(f"       full LP new_split for reference: "
          f"train={len(full['train'])} val={len(full['val'])} test={len(full['test'])}\n")

    # ── Assert canonical counts + hashes ────────────────────────────────────────
    ok = True
    for s, (exp_n, exp_h) in EXPECTED.items():
        pids = canon[s]
        got_n = len(pids)
        if got_n != exp_n:
            print(f"  [FAIL] {s}: count {got_n} != expected {exp_n}")
            ok = False
            continue
        if not args.no_hash:
            got_h = sha(pids)
            if got_h != exp_h:
                print(f"  [FAIL] {s}: count OK ({got_n}) but pid-set hash drifted")
                print(f"         expected {exp_h}")
                print(f"         got      {got_h}")
                ok = False
                continue
        print(f"  [ ok ] {s}: {got_n}{'  (hash matches)' if not args.no_hash else ''}")

    # Coords-only pool is reported as a soft check (not pinned by hash).
    for s, exp_n in COORDS_ONLY_EXPECTED.items():
        got = len(coords[s])
        if got != exp_n:
            print(f"  [warn] coords-only {s}: {got} != reference {exp_n} (atoms pool drifted)")

    if args.show_dropped and "filter_reason" in avail.columns:
        dropped = avail[avail.get("filtered", False).astype(bool)]
        print(f"\n  element-support filter dropped {len(dropped)} complexes:")
        for col in ("unsupported_pocket_elements", "unsupported_ligand_elements"):
            if col in avail.columns:
                vc = (dropped[col].dropna().astype(str)
                      .str.split(",").explode().str.strip().replace("", pd.NA).dropna().value_counts())
                if len(vc):
                    print(f"    {col}: " + ", ".join(f"{e}×{n}" for e, n in vc.head(12).items()))

    print()
    if ok:
        print("PASS — split reproduces the pinned canonical 2172/480/839 exactly.")
        return 0
    print("FAIL — split has drifted from the pinned canonical 2172/480/839.")
    print("       If this change is intentional, re-pin EXPECTED counts+hashes above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
