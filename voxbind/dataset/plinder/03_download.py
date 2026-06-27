"""03_download.py — Step 4: fetch raw structure + density into the SHARED caches.

Reads the FROZEN v2 selection (splits/plinder/v2/plinder_selected.csv, sha256-verified)
and, for each unique entry_pdb_id, downloads two artifacts from PDBe into the shared,
cross-dataset caches (resumable — files already present are skipped):

    data/ccp4/{pid}.ccp4    2Fo-Fc electron-density map   (SHARED; OTF reads this at train time)
    data/cif/{pid}.cif      deposited mmCIF (_updated)     (SHARED, new; carries label_asym_id)

Writes data/plinder/v2_acquire_manifest.json  {pid: {has_density, has_cif}} — the real
EDS hit-rate, so step 5 (build) can skip any PDB whose map 404s.

    cd voxbind && python dataset/plinder/03_download.py --limit 20   # smoke (first 20 pids)
    cd voxbind && python dataset/plinder/03_download.py              # full fetch
    cd voxbind && python dataset/plinder/03_download.py --what cifs  # only mmCIFs
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import pandas as pd

VOX_ROOT = Path(__file__).resolve().parents[2]            # .../voxbind
DATA = VOX_ROOT / "dataset" / "data"
CCP4_DIR = DATA / "ccp4"                                   # SHARED density cache (existing)
CIF_DIR = DATA / "cif"                                     # SHARED structure cache (new)
FREEZE = VOX_ROOT / "splits" / "plinder" / "v2"
MANIFEST = DATA / "plinder" / "v2_acquire_manifest.json"

# PDBe endpoints (same as legacy 03b — proven for PLINDER).
MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}_updated.cif"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# Reuse 00a's resumable `run_download` (thread pool; skips existing non-empty files; records 404s).
DL = load_module(VOX_ROOT / "dataset" / "00a_density_download.py", "dl00a")


def resolve_v2_selection() -> Path:
    """The frozen v2 CSV, sha256-verified against plinder_inputs.json (fail loudly on drift)."""
    csv = FREEZE / "plinder_selected.csv"
    if not csv.exists():
        raise SystemExit(f"[fatal] no frozen v2 selection at {csv} — run 01_select.py first.")
    inp = FREEZE / "plinder_inputs.json"
    if inp.exists():
        want = json.loads(inp.read_text()).get("selection_sha256")
        got = hashlib.sha256(csv.read_bytes()).hexdigest()
        if want and got != want:
            raise SystemExit(f"[fatal] v2 selection hash MISMATCH:\n  expected {want}\n  got      {got}")
        print(f"[selection] frozen v2 hash-verified: {csv}")
    return csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["both", "maps", "cifs"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="only the first N unique pids (0=all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    csv = resolve_v2_selection()
    pids = pd.read_csv(csv, usecols=["entry_pdb_id"]).entry_pdb_id.astype(str).str.lower().drop_duplicates().tolist()
    if args.limit:
        pids = pids[: args.limit]

    CCP4_DIR.mkdir(parents=True, exist_ok=True)
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[paths] ccp4 -> {CCP4_DIR}\n        cif  -> {CIF_DIR}")
    print(f"[03_download] {len(pids):,} unique pids" + (f" (limited to {args.limit})" if args.limit else ""))

    if args.what in ("both", "maps"):
        tasks = [(p, MAP_URL.format(pid=p), CCP4_DIR / f"{p}.ccp4") for p in pids]
        dl, have, fail, fails = DL.run_download(tasks, args.workers, "ccp4 maps")
        print(f"  maps: downloaded {dl}  already-had {have}  fail/404 {fail}")
    if args.what in ("both", "cifs"):
        tasks = [(p, CIF_URL.format(pid=p), CIF_DIR / f"{p}.cif") for p in pids]
        dl, have, fail, fails = DL.run_download(tasks, args.workers, "mmCIF")
        print(f"  cifs: downloaded {dl}  already-had {have}  fail/404 {fail}")

    cache = {p: {"has_density": (CCP4_DIR / f"{p}.ccp4").exists(),
                 "has_cif": (CIF_DIR / f"{p}.cif").exists()} for p in pids}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(cache, indent=2))
    n_both = sum(v["has_density"] and v["has_cif"] for v in cache.values())
    n_map = sum(v["has_density"] for v in cache.values())
    print(f"[done] {n_both:,}/{len(pids):,} have BOTH map+cif  ({n_map:,} have density)  -> {MANIFEST}")


if __name__ == "__main__":
    main()
