"""04_build.py — Step 5: build the v2 OTF corpus (role-split/diverse) from the SHARED caches.

Reuses 03c's proven cif parser (parse_cif_system, diverse=True → every heavy atom, per-atom
vdW radius, atoms_channel=0) but reads from the SHARED caches and writes v2-named outputs:

    in :  splits/plinder/v2/plinder_selected.csv   (hash-verified)
          data/cif/{pid}.cif                        (SHARED structures)
    out:  pretrain/data_train_plinder_v2.pt         (role-split tuples)
          pretrain/xray_resample_plinder_v2/        (OTF manifest + resample.json)

The fix vs 03c.run_diverse_build: resample.json's ccp4_dir points at SHARED data/ccp4
(03c hardcoded plinder/ccp4), so the download target and the train-time read path agree.
Pure OTF — no frozen crops. Laydown (shuffle 1234 → drop VAL_SZ → max_len filter) matches
DatasetCrossDockedDensity so manifest row i aligns with dataset index i.

    cd voxbind && python dataset/plinder/04_build.py --limit 50   # smoke
    cd voxbind && python dataset/plinder/04_build.py              # full build
"""
import gemmi  # noqa: F401 — before torch (shared-lib load order)
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, hashlib, importlib.util, json, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
torch.set_num_threads(2)

VOX_ROOT = Path(__file__).resolve().parents[2]
DATA = VOX_ROOT / "dataset" / "data"
CIF_DIR = DATA / "cif"                       # SHARED structures
CCP4_DIR = DATA / "ccp4"                      # SHARED density (OTF reads at train time)
PRETRAIN = DATA / "pretrain"
FREEZE = VOX_ROOT / "splits" / "plinder" / "v2"
OUT_TUPLES = PRETRAIN / "data_train_plinder_v2.pt"
OUT_DIR = PRETRAIN / "xray_resample_plinder_v2"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


M = load_module(VOX_ROOT / "dataset" / "legacy" / "03c_plinder_preprocess.py", "p03c")  # parser + constants


def resolve_v2_selection() -> Path:
    csv = FREEZE / "plinder_selected.csv"
    if not csv.exists():
        raise SystemExit(f"[fatal] no frozen v2 selection at {csv} — run 01_select.py first.")
    inp = FREEZE / "plinder_inputs.json"
    if inp.exists():
        want = json.loads(inp.read_text()).get("selection_sha256")
        got = hashlib.sha256(csv.read_bytes()).hexdigest()
        if want and got != want:
            raise SystemExit(f"[fatal] v2 selection hash MISMATCH:\n  expected {want}\n  got {got}")
    return csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N rows (smoke)")
    ap.add_argument("--pocket_radius", type=float, default=10.0)
    args = ap.parse_args()

    csv = resolve_v2_selection()
    sel = pd.read_csv(csv)
    if args.limit:
        sel = sel.head(args.limit)
    print(f"=== v2 DIVERSE build (role-split, OTF) ===\n  selection: {csv} ({len(sel):,} rows)\n  cifs: {CIF_DIR}")

    tuples, skipped = [], {}
    for r in tqdm(list(sel.itertuples(index=False)), desc="build", unit="cplx"):
        pid = str(r.entry_pdb_id).lower()
        key = f"{pid}_{r.ligand_asym_id}"
        cif = CIF_DIR / f"{pid}.cif"
        if not cif.exists():
            skipped[key] = "missing_cif"; continue
        try:
            parsed = M.parse_cif_system(cif, r.ligand_asym_id, r.ligand_ccd_code, args.pocket_radius, diverse=True)
        except Exception as e:
            skipped[key] = f"error:{type(e).__name__}"; continue
        if parsed is None:
            skipped[key] = "parse_failed"; continue
        lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, lig_rad, poc_rad = parsed
        max_len = round(float((lig_xyz.max(0).values - lig_xyz.min(0).values).max()), 2)
        pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
                   "atoms_channel": poc_ch.to(torch.uint8), "radius": poc_rad.to(torch.float32)}
        ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
                   "atoms_channel": lig_ch.to(torch.uint8), "radius": lig_rad.to(torch.float32),
                   "max_len": max_len}
        tuples.append((pocket_, ligand_))

    print(f"  built {len(tuples):,} tuples  (skipped {len(skipped):,})")
    for reason, n in Counter(skipped.values()).most_common():
        print(f"     skip {reason:20s} {n}")
    if not tuples:
        raise SystemExit("  [abort] no tuples built.")

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(OUT_TUPLES))
    print(f"  {OUT_TUPLES.name}: {len(tuples):,} tuples")

    # normalization recipe: reuse the existing PLINDER v6 arcsinh+z stats (byte-identical scale)
    src = PRETRAIN / "xray_resample_plinder" / "resample.json"
    norm_block = json.loads(src.read_text())["normalization"]

    # laydown mirrors DatasetCrossDockedDensity: shuffle(1234) → drop VAL_SZ → max_len filter
    order = list(tuples)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - M.VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= M.MAX_LEN]
    n = len(order)
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = (np.stack([l["coords"].to(torch.float32).mean(0).numpy() for _p, l in order]).astype(np.float32)
                 if n else np.zeros((0, 3), np.float32))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "train_manifest.npz",
             pdb_id=np.array(pdb_ids), centroid=centroids,
             R=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(),
             t=np.zeros((n, 3), dtype=np.float32), ok=np.ones(n, dtype=bool))
    np.save(str(OUT_DIR / "train_available.npy"), np.ones(n, dtype=bool))
    (OUT_DIR / "resample.json").write_text(json.dumps(dict(
        kind="resample_manifest", grid_dim=M.GRID_DIM, resolution=M.RESOLUTION,
        ccp4_dir=str(CCP4_DIR.resolve()), ccp4_ext=".ccp4",          # ← SHARED cache (the v2 fix)
        frame="deposited (transform=None): R=I, t=0", norm_version="v6", normalization=norm_block,
        train_time_note="v2 DIVERSE role-split (data_train_plinder_v2.pt): per-atom vdW radius, "
                        "atoms_channel=0 → single role channel admits all elements; reads SHARED data/ccp4.",
    ), indent=2))
    print(f"  manifest: {n:,} train positions → {OUT_DIR/'train_manifest.npz'}")
    print(f"  launch: dset.resample_dir={OUT_DIR} dset.data_file=data_train_plinder_v2.pt "
          f"dset.subset_n={max(n - M.VAL_SZ, 0)} dset.subset_val_n={M.VAL_SZ} dset.subset_xray_only=true "
          f"dset.ligand_radius=-1 dset.pocket_radius=-1")


if __name__ == "__main__":
    main()
