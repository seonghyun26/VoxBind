"""03a_plinder_select.py — Select PLINDER ligand-instances for density pretraining.

PLINDER (gs://plinder/2024-06/v2) is used purely as a CURATED SELECTION TABLE.
We read the annotation table + split parquet, apply quality / ligand-type /
density filters, drop anything overlapping our held-out downstream test sets,
dedup, and emit a CSV of (entry_pdb_id, ligand_asym_id, ligand_ccd_code, ...).
The heavy data (deposited mmCIF + EDS density map) is pulled later by 03b from
PDBe, keeping everything in the *deposited crystal frame* the density map lives
in (Strategy A: transform=None, like the PDBbind path in 01b/00h).

Why PLINDER is a cleaner density source than CrossDocked/PDBbind
---------------------------------------------------------------
The annotation table carries per-system real-space density-fit validation:
    system_ligand_validation_average_rscc   real-space correlation of the ligand
                                            to its experimental 2Fo-Fc density
    system_ligand_validation_average_occupancy
    ligand_num_unresolved_heavy_atoms       ligand heavy atoms with no density
Requiring RSCC >= 0.8 and 0 unresolved atoms *guarantees the density actually
corresponds to the selected ligand* — the cleanest possible ligand-matched
density (cf. CrossDocked, where 86% of poses are cross-docked into a pocket whose
density belongs to a different ligand; see project_density_v6_ligand_matched).

Leakage
-------
PLINDER's own split protects PLINDER's benchmark, not ours. We additionally drop
every case whose 4-char entry_pdb_id appears in any of our downstream test sets:
    PDBbind index.csv new_split in {test, val}   (affinity / probe eval)
    MISATO pool (dataset/data/misato/pool_pids.txt)   (QM / MD eval)
    CrossDocked data_test.pt                      (generation eval)

Outputs (under dataset/data/plinder/)
-------------------------------------
    index/annotation_table.parquet   downloaded if missing (~1 GB)
    splits/split.parquet             downloaded if missing (~11 MB)
    plinder_selected.csv             the selection (one row per kept ligand-instance)
    plinder_funnel.json              per-stage counts (reproducibility / reporting)

Usage
-----
    cd voxbind
    python dataset/03a_plinder_select.py                       # full selection (~27k)
    python dataset/03a_plinder_select.py --n 1500 --seed 42    # smoke pilot subset
    python dataset/03a_plinder_select.py --min_rscc 0.85 --max_res 2.0   # stricter
    python dataset/03a_plinder_select.py --dedup ecod          # dedup by pocket fold

GCS layout (public, anonymous read):
    https://storage.googleapis.com/plinder/2024-06/v2/index/annotation_table.parquet
    https://storage.googleapis.com/plinder/2024-06/v2/splits/split.parquet
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

VOXDATA = Path(__file__).resolve().parent.parent / "data"
PLINDER = VOXDATA / "plinder"

GCS = "https://storage.googleapis.com/plinder/2024-06/v2"
ANNOT_URL = f"{GCS}/index/annotation_table.parquet"
SPLIT_URL = f"{GCS}/splits/split.parquet"

# Ligand heavy-element vocabulary the voxelizer supports (constants.ELEMENTS_HASH).
LIGAND_VOCAB = {"C", "O", "N", "S", "F", "Cl", "P"}

# Annotation columns we actually read (projection keeps the 1.3M x 743 table cheap).
ANNOT_COLS = [
    "entry_pdb_id", "system_id", "system_biounit_id",
    "entry_determination_method", "entry_resolution",
    "ligand_is_proper", "ligand_is_ion", "ligand_is_artifact", "ligand_is_cofactor",
    "ligand_is_covalent", "ligand_is_invalid", "ligand_is_rdkit_loadable",
    "ligand_num_heavy_atoms", "ligand_num_unresolved_heavy_atoms",
    "ligand_molecular_weight",
    "system_ligand_validation_average_rscc",
    "system_ligand_validation_average_occupancy",
    "ligand_ccd_code", "ligand_asym_id",
    "ligand_rdkit_canonical_smiles",
    "system_num_ligand_chains", "system_proper_num_ligand_chains",
    "system_pocket_ECOD", "system_pocket_UniProt",
]


# ── downloads ────────────────────────────────────────────────────────────────────

def fetch(url: str, dest: Path) -> None:
    """Download `url` -> `dest` if not already present (atomic via .tmp)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {dest.name} from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.0f} MB)")


# ── downstream-test exclusion set ─────────────────────────────────────────────────

_CROSSDOCKED_ID_RE = re.compile(
    r"(?:^|/)(?P<receptor>[0-9][A-Za-z0-9]{3})_[^_/]+_rec_"
    r"(?P<ligand_source>[0-9][A-Za-z0-9]{3})_",
    re.IGNORECASE,
)


def crossdocked_test_pdb_ids(data) -> set[str]:
    """Extract real receptor + ligand-source PDB IDs, not target-family prefixes."""
    ids: set[str] = set()
    for index, (pocket, ligand) in enumerate(data):
        matches = [
            _CROSSDOCKED_ID_RE.search(str(record["id"]))
            for record in (pocket, ligand)
        ]
        if any(match is None for match in matches):
            raise ValueError(f"cannot parse CrossDocked test row {index}")
        parsed = [
            (match.group("receptor").lower(), match.group("ligand_source").lower())
            for match in matches
        ]
        if parsed[0] != parsed[1]:
            raise ValueError(f"CrossDocked test row {index} has mismatched IDs: {parsed}")
        ids.update(parsed[0])
    return ids


def downstream_test_pdb_ids() -> set:
    """4-char lowercase PDB ids held out by any downstream eval (never pretrain on)."""
    excl: set = set()

    idx = VOXDATA / "pdbbind" / "index.csv"
    if idx.exists():
        df = pd.read_csv(idx)
        held = df.loc[df.new_split.isin(["test", "val"]), "pdb_id"].astype(str).str.lower()
        excl |= set(held)
        print(f"  pdbbind test+val: +{len(set(held))}")

    pool = VOXDATA / "misato" / "pool_pids.txt"
    if pool.exists():
        ids = {l.strip().lower() for l in pool.read_text().splitlines() if l.strip()}
        excl |= ids
        print(f"  misato pool:      +{len(ids)}")

    cd = VOXDATA / "data_test.pt"
    if cd.exists():
        data = torch.load(cd, weights_only=False)
        ids = crossdocked_test_pdb_ids(data)
        excl |= ids
        print(f"  crossdocked test: +{len(ids)}")

    print(f"  exclusion set: {len(excl)} unique pdb ids")
    return excl


# ── selection funnel ──────────────────────────────────────────────────────────────

def select(df: pd.DataFrame, args, excl: set):
    """Apply the staged filter funnel; return (selection_df, funnel_list)."""
    res = pd.to_numeric(df.entry_resolution, errors="coerce")
    rscc = pd.to_numeric(df.system_ligand_validation_average_rscc, errors="coerce")
    nh = pd.to_numeric(df.ligand_num_heavy_atoms, errors="coerce")
    unres = pd.to_numeric(df.ligand_num_unresolved_heavy_atoms, errors="coerce")

    def b(col):  # nullable-bool -> strict True
        return df[col].eq(True)

    stages = [
        ("all", pd.Series(True, index=df.index)),
        ("X-ray", df.entry_determination_method.eq("X-RAY DIFFRACTION")),
        (f"resolution<={args.max_res}", res.le(args.max_res)),
        ("proper & !ion/artifact/cofactor",
         b("ligand_is_proper") & ~b("ligand_is_ion") & ~b("ligand_is_artifact") & ~b("ligand_is_cofactor")),
        ("!invalid & rdkit_loadable", ~b("ligand_is_invalid") & b("ligand_is_rdkit_loadable")),
        (f"{args.min_heavy}<=heavy<={args.max_heavy}", nh.ge(args.min_heavy) & nh.le(args.max_heavy)),
        ("unresolved_heavy==0", unres.eq(0)),
        (f"rscc>={args.min_rscc}", rscc.ge(args.min_rscc)),
    ]
    if not args.allow_covalent:
        stages.append(("!covalent", ~b("ligand_is_covalent")))
    if args.single_ligand:
        stages.append(("single proper ligand", df.system_proper_num_ligand_chains.eq(1)))
    if args.plinder_splits:
        stages.append((f"plinder split in {args.plinder_splits}", df.split.isin(args.plinder_splits)))
    stages.append(("exclude downstream-test", ~df.entry_pdb_id.str.lower().isin(excl)))

    funnel, mask = [], pd.Series(True, index=df.index)
    for name, cond in stages:
        mask &= cond.reindex(df.index).fillna(False)
        rows, pdbs = int(mask.sum()), int(df.loc[mask, "entry_pdb_id"].nunique())
        funnel.append({"stage": name, "ligand_instances": rows, "unique_pdb": pdbs})
        print(f"  {name:34s} rows={rows:8d}  uniq_pdb={pdbs:7d}")

    sel = df[mask].copy()
    # one row per physical ligand instance, best-resolved first
    sel["_rscc"] = pd.to_numeric(sel.system_ligand_validation_average_rscc, errors="coerce")
    sel = (sel.sort_values("_rscc", ascending=False)
              .drop_duplicates(["entry_pdb_id", "ligand_asym_id"]))

    dedup_key = {"pdb": ["entry_pdb_id"],
                 "pdb_ccd": ["entry_pdb_id", "ligand_ccd_code"],
                 "ecod": ["system_pocket_ECOD"],
                 "none": None}[args.dedup]
    if dedup_key:
        before = len(sel)
        # keep best-RSCC representative per group; ECOD nan -> keep all (unclustered)
        if args.dedup == "ecod":
            has = sel.system_pocket_ECOD.notna() & sel.system_pocket_ECOD.ne("")
            sel = pd.concat([sel[has].drop_duplicates(dedup_key), sel[~has]])
        else:
            sel = sel.drop_duplicates(dedup_key)
        print(f"  dedup[{args.dedup}]: {before} -> {len(sel)}")

    return sel.reset_index(drop=True), funnel


# ── CLI ────────────────────────────────────────────────────────────────────────

def _freeze_to_splits(sel_csv: Path, funnel_json: Path, args) -> None:
    """Re-pin the committed frozen selection (voxbind/splits/plinder/) from a fresh 03a build:
    copy the selection CSV + funnel and (re)write plinder_inputs.json with the new sha256 +
    leakage-file hashes. Consumed by 03b/03c via voxbind.splits.frozen_plinder_selection()."""
    import hashlib
    import shutil
    dst = Path(__file__).resolve().parents[2] / "splits" / "plinder"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sel_csv, dst / "plinder_selected.csv")
    if funnel_json.exists():
        shutil.copy2(funnel_json, dst / "plinder_funnel.json")

    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None

    n = sum(1 for _ in open(dst / "plinder_selected.csv")) - 1
    inputs = {
        "name": "plinder_v1",
        "description": "Frozen PLINDER ligand-matched density pretraining selection (cross-server stable).",
        "plinder_bucket": GCS,
        "selection_csv": "plinder_selected.csv",
        "selection_sha256": _sha(dst / "plinder_selected.csv"),
        "n_selected": n,
        "filters": {k: getattr(args, k, None) for k in
                    ("min_rscc", "max_res", "min_heavy", "max_heavy",
                     "single_ligand", "allow_covalent", "plinder_splits", "dedup")},
        "build": {"shuffle_seed": 1234, "val_tail": 100, "max_len": 30,
                  "norm_version": "v6_arcsinh_z", "frame": "deposited (transform=None)",
                  "pocket_radius": 10.0},
        "leakage_holdout": {
            "pdbbind_index_csv":   {"path": "dataset/data/pdbbind/index.csv",    "sha256": _sha(VOXDATA / "pdbbind" / "index.csv")},
            "misato_pool_pids":    {"path": "dataset/data/misato/pool_pids.txt", "sha256": _sha(VOXDATA / "misato" / "pool_pids.txt")},
            "crossdocked_test_pt": {"path": "dataset/data/data_test.pt",         "sha256": _sha(VOXDATA / "data_test.pt")},
        },
        "note": "Re-pinned via 03a --freeze. Consumed by 03b/03c via voxbind.splits.frozen_plinder_selection().",
    }
    (dst / "plinder_inputs.json").write_text(json.dumps(inputs, indent=2))
    gi = dst / ".gitignore"
    if not gi.exists():
        gi.write_text("# repo-wide .gitignore drops *.csv/*.json; re-include the frozen selection here.\n"
                      "!*.csv\n!*.json\n!*.md\n!.gitignore\n")
    print(f"[freeze] re-pinned committed selection -> {dst}  "
          f"(sha {inputs['selection_sha256'][:12]}…, n={n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_res", type=float, default=2.5, help="max resolution (A)")
    ap.add_argument("--min_rscc", type=float, default=0.80, help="min ligand real-space CC to its density")
    ap.add_argument("--min_heavy", type=int, default=6)
    ap.add_argument("--max_heavy", type=int, default=50)
    ap.add_argument("--single_ligand", action=argparse.BooleanOptionalAction, default=True,
                    help="require exactly one proper ligand chain per system")
    ap.add_argument("--allow_covalent", action="store_true", help="keep covalent ligands")
    ap.add_argument("--plinder_splits", nargs="*", default=["train"],
                    help="PLINDER split labels to keep (train/val/test); [] = ignore split")
    ap.add_argument("--dedup", choices=["pdb", "pdb_ccd", "ecod", "none"], default="pdb",
                    help="redundancy reduction (one representative per group)")
    ap.add_argument("--n", type=int, default=0, help="subsample N rows (0 = keep all); for the pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PLINDER / "plinder_selected.csv"))
    ap.add_argument("--rederive", action="store_true",
                    help="explicit acknowledgement that you are RE-deriving the selection from the "
                         "(mutable) annotation table; pair with --freeze to re-pin the committed copy.")
    ap.add_argument("--freeze", action="store_true",
                    help="after building, RE-PIN the committed frozen selection in splits/plinder/ "
                         "(copy CSV + funnel, recompute sha256 + leakage hashes in plinder_inputs.json). "
                         "This is the deliberate, version-bumped re-pin step.")
    args = ap.parse_args()

    print("[1] ensure parquet inputs")
    fetch(ANNOT_URL, PLINDER / "index" / "annotation_table.parquet")
    fetch(SPLIT_URL, PLINDER / "splits" / "split.parquet")

    print("[2] downstream-test exclusion set")
    excl = downstream_test_pdb_ids()

    print("[3] load annotation (projected cols) + split")
    df = pq.read_table(PLINDER / "index" / "annotation_table.parquet", columns=ANNOT_COLS).to_pandas()
    print(f"  annotation: {len(df):,} ligand-instances, {df.entry_pdb_id.nunique():,} pdb ids")
    spl = pq.read_table(PLINDER / "splits" / "split.parquet").to_pandas()
    scol = "split" if "split" in spl.columns else [c for c in spl.columns if "split" in c.lower()][0]
    key = "system_id" if "system_id" in spl.columns else [c for c in spl.columns if c.lower().endswith("system_id")][0]
    smap = dict(zip(spl[key].astype(str), spl[scol].astype(str)))
    df["split"] = df.system_id.astype(str).map(smap).fillna("none")
    print(f"  split join on {key!r}/{scol!r}: " + ", ".join(f"{k}={v}" for k, v in df.split.value_counts().items()))

    print("[4] selection funnel")
    sel, funnel = select(df, args, excl)

    if args.n and len(sel) > args.n:
        sel = sel.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"  subsample -> {len(sel)} rows (seed={args.seed})")

    keep = ["entry_pdb_id", "system_id", "system_biounit_id", "ligand_asym_id", "ligand_ccd_code",
            "ligand_rdkit_canonical_smiles", "entry_resolution",
            "system_ligand_validation_average_rscc", "system_ligand_validation_average_occupancy",
            "ligand_num_heavy_atoms", "ligand_molecular_weight",
            "system_pocket_ECOD", "system_pocket_UniProt", "split"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sel[keep].to_csv(out, index=False)
    (PLINDER / "plinder_funnel.json").write_text(json.dumps(
        {"args": vars(args), "funnel": funnel,
         "n_selected": int(len(sel)), "n_unique_pdb": int(sel.entry_pdb_id.nunique())}, indent=2))

    if args.freeze:
        _freeze_to_splits(out, PLINDER / "plinder_funnel.json", args)

    print(f"\n[done] wrote {len(sel)} rows -> {out}")
    print(f"       unique pdb ids: {sel.entry_pdb_id.nunique()}")
    print(f"       resolution: median {pd.to_numeric(sel.entry_resolution).median():.2f} A")
    print(f"       RSCC: median {sel['_rscc'].median() if '_rscc' in sel else float('nan'):.3f}"
          if "_rscc" in sel.columns else "")


if __name__ == "__main__":
    main()
