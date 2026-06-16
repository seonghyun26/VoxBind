"""03b_plinder_acquire.py — Download deposited structure + density for PLINDER cases.

Consumes dataset/data/plinder/plinder_selected.csv (from 03a). For each unique
entry_pdb_id it fetches two artifacts from PDBe, mirroring 00a's resumable
thread-pool downloader (reused verbatim):

    dataset/data/plinder/ccp4/{pid}.ccp4   2Fo-Fc electron-density map  (the density signal)
    dataset/data/plinder/cif/{pid}.cif     deposited mmCIF (updated)    (atom coords + label_asym_id)

Why the mmCIF (vs 00a's legacy .pdb): PLINDER identifies each ligand instance by
its label_asym_id (the `ligand_asym_id` column). The PDBe *_updated.cif carries
label_asym_id, lets 03c pick out exactly that ligand + its pocket, and is in the
deposited crystal frame the .ccp4 map lives in — so 03c crops with transform=None
(Strategy A; no Kabsch needed, unlike CrossDocked's 00b).

EDS coverage is effectively complete for our selection: 03a already required
system_ligand_validation_average_rscc >= 0.8, which presupposes deposited
structure factors → an EDS map exists. A coverage cache is still written so 03c
can skip any straggler that 404s, and so we can report the real hit rate.

Outputs:
    dataset/data/plinder/ccp4/{pid}.ccp4
    dataset/data/plinder/cif/{pid}.cif
    dataset/data/plinder/acquire_cache.json   {pid: {has_density, has_cif}}

Usage
-----
    cd voxbind
    python dataset/03b_plinder_acquire.py                 # all unique pids in the selection
    python dataset/03b_plinder_acquire.py --limit 1500    # smoke pilot (first N pids)
    python dataset/03b_plinder_acquire.py --workers 32 --what maps
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

VOX_ROOT = Path(__file__).resolve().parents[2]
VOXDATA = VOX_ROOT / "dataset" / "data"
PLINDER = VOXDATA / "plinder"

# PDBe endpoints (same host family as 00a's MAP_URL / PDB_URL).
MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}_updated.cif"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse 00a's resumable `download` + thread-pool `run_download` verbatim.
DL = load_module(Path(__file__).parent.parent / "00a_density_download.py", "dl00a")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--selection", default=str(PLINDER / "plinder_selected.csv"))
    ap.add_argument("--what", choices=["both", "maps", "cifs"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="only the first N unique pids (pilot)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.selection)
    pids = df.entry_pdb_id.astype(str).str.lower().drop_duplicates().tolist()
    if args.limit:
        pids = pids[: args.limit]
    print(f"[03b] {len(pids)} unique pids from {Path(args.selection).name}"
          + (f" (limited to {args.limit})" if args.limit else ""))

    ccp4_dir = PLINDER / "ccp4"
    cif_dir = PLINDER / "cif"
    ccp4_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)

    failures = {}
    if args.what in ("both", "maps"):
        tasks = [(p, MAP_URL.format(pid=p), ccp4_dir / f"{p}.ccp4") for p in pids]
        dl, have, fail, fails = DL.run_download(tasks, args.workers, "ccp4 maps")
        print(f"  maps: downloaded {dl}  have {have}  fail {fail}")
        failures["maps"] = dict(fails)
    if args.what in ("both", "cifs"):
        tasks = [(p, CIF_URL.format(pid=p), cif_dir / f"{p}.cif") for p in pids]
        dl, have, fail, fails = DL.run_download(tasks, args.workers, "mmCIF")
        print(f"  cifs: downloaded {dl}  have {have}  fail {fail}")
        failures["cifs"] = dict(fails)

    # Coverage cache: which pids actually have each artifact on disk now.
    cache = {p: {"has_density": (ccp4_dir / f"{p}.ccp4").exists(),
                 "has_cif": (cif_dir / f"{p}.cif").exists()} for p in pids}
    (PLINDER / "acquire_cache.json").write_text(json.dumps(cache, indent=2))
    n_both = sum(v["has_density"] and v["has_cif"] for v in cache.values())
    print(f"[done] {n_both}/{len(pids)} pids have BOTH map+cif  "
          f"-> {PLINDER/'acquire_cache.json'}")
    if any(failures.values()):
        miss = [p for p, v in cache.items() if not (v["has_density"] and v["has_cif"])]
        print(f"  {len(miss)} incomplete (e.g. {miss[:8]})")


if __name__ == "__main__":
    main()
