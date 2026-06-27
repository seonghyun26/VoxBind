"""01_select.py — PLINDER v2 selection (robust pipeline, steps 1-3).

Step 1  COMPLEX filter   — is this a real, representable protein+ligand complex?
Step 2  DENSITY filter   — does the experimental density actually match the ligand?
Step 3  FREEZE CSV       — pin the surviving complexes to splits/plinder/v2/ with a
                           sha256 so every server builds from the same list.

Selection is metadata-only (PLINDER annotation table) — NO downloads here. The raw
mmCIF + 2Fo-Fc maps are fetched later (step 4, 03_download) into the SHARED caches
data/cif + data/ccp4, and cropped (step 5) into the PLINDER-specific corpus.

v2 recipe (vs v1): drop in-vocab + resolution filters (RSCC carries quality),
all PLINDER splits, multi-ligand, no dedup. ~113,976 instances / ~41,833 PDB.

    cd voxbind && python dataset/plinder/01_select.py            # build + freeze v2
    cd voxbind && python dataset/plinder/01_select.py --dry_run  # funnel only, no write
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, hashlib, json
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
import torch; torch.set_num_threads(2)

VOX = Path(__file__).resolve().parents[2]            # .../voxbind
DATA = VOX / "dataset" / "data"
PLINDER = DATA / "plinder"
FREEZE = VOX / "splits" / "plinder" / "v2"

RECIPE = dict(min_rscc=0.80, max_res=0, min_heavy=6, max_heavy=50,   # max_res=0 -> no cap
              single_ligand=False, allow_covalent=False, plinder_splits=[],  # [] = all splits
              dedup="none", in_vocab_filter=False)
BUILD = dict(shuffle_seed=1234, val_tail=100, max_len=30,
             norm_version="v6_arcsinh_z", pocket_radius=10.0, frame="deposited (transform=None)")

KEEP = ["entry_pdb_id", "system_id", "system_biounit_id", "ligand_asym_id", "ligand_ccd_code",
        "ligand_rdkit_canonical_smiles", "entry_resolution",
        "system_ligand_validation_average_rscc", "system_ligand_validation_average_occupancy",
        "ligand_num_heavy_atoms", "ligand_molecular_weight",
        "system_pocket_ECOD", "system_pocket_UniProt", "split"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def leakage_holdout():
    """4-char lowercase PDB ids held out by any downstream eval, + their file hashes."""
    excl, hashes = set(), {}
    idx = DATA / "pdbbind" / "index.csv"
    if idx.exists():
        d = pd.read_csv(idx)
        excl |= set(d.loc[d.new_split.isin(["test", "val"]), "pdb_id"].astype(str).str.lower())
        hashes["pdbbind_index_csv"] = {"path": str(idx.relative_to(VOX)), "sha256": sha256(idx)}
    pool = DATA / "misato" / "pool_pids.txt"
    if pool.exists():
        excl |= {l.strip().lower() for l in pool.read_text().splitlines() if l.strip()}
        hashes["misato_pool_pids"] = {"path": str(pool.relative_to(VOX)), "sha256": sha256(pool)}
    for cd in [DATA / "data_test.pt", DATA / "pretrain" / "data_test.pt"]:
        if cd.exists():
            data = torch.load(cd, weights_only=False)
            excl |= {str(p[0]["id"])[:4].lower() for p in data}
            hashes["crossdocked_test_pt"] = {"path": str(cd.relative_to(VOX)), "sha256": sha256(cd)}
            break
    return excl, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="print funnel, do not write the freeze")
    args = ap.parse_args()

    excl, leak_hashes = leakage_holdout()
    print(f"[leakage] {len(excl):,} held-out PDB ids\n")

    cols = list(dict.fromkeys(KEEP[:-1] + [
        "entry_determination_method", "ligand_is_proper", "ligand_is_ion", "ligand_is_artifact",
        "ligand_is_cofactor", "ligand_is_covalent", "ligand_is_invalid", "ligand_is_rdkit_loadable",
        "ligand_num_unresolved_heavy_atoms"]))
    df = pq.read_table(PLINDER / "index" / "annotation_table.parquet", columns=cols).to_pandas()
    spl = pq.read_table(PLINDER / "splits" / "split.parquet").to_pandas()
    scol = "split" if "split" in spl.columns else [c for c in spl.columns if "split" in c.lower()][0]
    key = "system_id" if "system_id" in spl.columns else [c for c in spl.columns if c.lower().endswith("system_id")][0]
    df["split"] = df.system_id.astype(str).map(dict(zip(spl[key].astype(str), spl[scol].astype(str)))).fillna("none")

    res = pd.to_numeric(df.entry_resolution, errors="coerce")
    rscc = pd.to_numeric(df.system_ligand_validation_average_rscc, errors="coerce")
    nh = pd.to_numeric(df.ligand_num_heavy_atoms, errors="coerce")
    un = pd.to_numeric(df.ligand_num_unresolved_heavy_atoms, errors="coerce")
    b = lambda c: df[c].eq(True)

    stages = [
        ("STEP 1 — COMPLEX filter", [
            ("all",                              pd.Series(True, index=df.index)),
            ("X-ray diffraction",                df.entry_determination_method.eq("X-RAY DIFFRACTION")),
            ("proper & !ion/artifact/cofactor",  b("ligand_is_proper") & ~b("ligand_is_ion") & ~b("ligand_is_artifact") & ~b("ligand_is_cofactor")),
            ("!invalid & rdkit_loadable",        ~b("ligand_is_invalid") & b("ligand_is_rdkit_loadable")),
            (f"{RECIPE['min_heavy']} <= heavy <= {RECIPE['max_heavy']}", nh.ge(RECIPE['min_heavy']) & nh.le(RECIPE['max_heavy'])),
            ("!covalent",                        ~b("ligand_is_covalent")),
            ("exclude downstream-test",          ~df.entry_pdb_id.str.lower().isin(excl)),
        ]),
        ("STEP 2 — DENSITY filter", [
            ("unresolved_heavy == 0",            un.eq(0)),
            (f"RSCC >= {RECIPE['min_rscc']}",    rscc.ge(RECIPE['min_rscc'])),
        ]),
    ]
    print(f"{'stage':<40}{'inst':>12}{'removed':>11}{'uniq_pdb':>11}")
    print("-" * 74)
    mask = pd.Series(True, index=df.index); prev = None; funnel = []
    for group, subs in stages:
        print(group)
        for name, cond in subs:
            mask = mask & cond.reindex(df.index).fillna(False)
            ni = int(mask.sum()); npdb = int(df.loc[mask, "entry_pdb_id"].nunique())
            rem = "" if prev is None else f"{ni - prev:+,}"
            print(f"  {name:<38}{ni:>12,}{rem:>11}{npdb:>11,}")
            funnel.append({"stage": name, "ligand_instances": ni, "unique_pdb": npdb}); prev = ni

    sel = df.loc[mask, KEEP].reset_index(drop=True)
    n, npdb, nccd = len(sel), sel.entry_pdb_id.nunique(), sel.ligand_ccd_code.nunique()
    print(f"\nv2 selection: {n:,} instances / {npdb:,} PDB / {nccd:,} chemotypes")

    if args.dry_run:
        print("\n[dry_run] nothing written."); return

    FREEZE.mkdir(parents=True, exist_ok=True)
    csv = FREEZE / "plinder_selected.csv"
    sel.to_csv(csv, index=False)
    (FREEZE / "plinder_funnel.json").write_text(json.dumps(
        {"funnel": funnel, "n_selected": int(n), "n_unique_pdb": int(npdb), "n_chemotypes": int(nccd)}, indent=2))
    inputs = {
        "name": "plinder_v2",
        "description": "Frozen PLINDER v2 ligand-matched density pretraining selection (max-coverage; RSCC-gated).",
        "plinder_bucket": "gs://plinder/2024-06/v2",
        "selection_csv": "plinder_selected.csv",
        "selection_sha256": sha256(csv),
        "n_selected": int(n),
        "filters": RECIPE,
        "build": BUILD,
        "leakage_holdout": leak_hashes,
        "note": "Built via dataset/plinder/ pipeline. Downloads -> shared data/cif + data/ccp4; "
                "crops -> pretrain/xray_crops_aligned_plinder_v2. Re-pin deliberately (version-bumped).",
    }
    (FREEZE / "plinder_inputs.json").write_text(json.dumps(inputs, indent=2))
    print(f"\n[frozen] {FREEZE}/  (selected.csv sha256 {inputs['selection_sha256'][:12]}…)")


if __name__ == "__main__":
    main()
