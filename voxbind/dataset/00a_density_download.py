"""00a_density_download.py — Download X-ray data for CrossDocked2020 density crops.

Two artifacts per PDB ID, both fetched from PDBe entry-files and consumed
downstream by dataset/00b_density_preprocess.py:

    dataset/data/ccp4/{pdb}.ccp4   2Fo-Fc electron-density map  (CCP4 / MRC)
    dataset/data/pdb/{pdb}.pdb     deposited coordinates        (for Kabsch frame alignment)

The map is the density signal; the deposited PDB is needed by 00b's alignment
stage to recover the CrossDocked-frame → crystal-frame rigid transform.

PDB IDs are derived from the dataset splits themselves (data_train.pt /
data_test.pt, same max_len filter as the dataset class) so the download covers
exactly what 00b will need. An optional EDS availability cache restricts the
*map* downloads to PDB IDs with confirmed PDBe-EDS structure factors (skips
known 404s); deposited PDBs are attempted for every derived ID.

Both downloads are resumable (existing non-empty files are skipped) and run on a
thread pool. Maps that 404 (no EDS map for that PDB) are recorded, not retried.

Usage
-----
    cd voxbind
    python dataset/00a_density_download.py                      # maps + pdbs, train+test
    python dataset/00a_density_download.py --what maps          # only CCP4 maps
    python dataset/00a_density_download.py --what pdbs          # only deposited PDBs
    python dataset/00a_density_download.py \
        --eds_cache ../notebook/data/check_xray_cache.json      # restrict maps to confirmed EDS
    python dataset/00a_density_download.py --workers 32

URLs
----
    map : https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4
    pdb : https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from tqdm import tqdm

VOXDATA = Path(__file__).resolve().parent / "data"

MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
PDB_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent"
MAX_RETRIES = 3
RETRY_WAIT = 2.0


# ── PDB-ID discovery (mirrors DatasetCrossDockedXray / 00b load order) ──────────

def load_split(data_dir: str, split: str, max_len: int):
    if split in ("train", "val"):
        data = torch.load(os.path.join(data_dir, "data_train.pt"), weights_only=False)
        random.Random(1234).shuffle(data)
        val_sz = 100
        data = data[: len(data) - val_sz] if split == "train" else data[len(data) - val_sz :]
    else:
        data = torch.load(os.path.join(data_dir, "data_test.pt"), weights_only=False)
    return [(p, l) for p, l in data if l["max_len"] <= max_len]


def get_pdb_id(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].lower()


def pdb_ids_from_splits(data_dir: str, splits: list, max_len: int) -> list:
    """Sorted union of PDB IDs referenced by the requested splits."""
    ids: set = set()
    for split in splits:
        for pocket_, _ in load_split(data_dir, split, max_len):
            ids.add(get_pdb_id(pocket_["id"]))
    return sorted(ids)


def load_eds_pdb_ids(eds_cache_path: Path) -> set:
    """PDB IDs with confirmed PDBe-EDS X-ray density (lowercase)."""
    cache = json.loads(Path(eds_cache_path).read_text())
    return {
        info["pdb_id"].lower()
        for info in cache.values()
        if info.get("has_density") and info.get("source") == "PDBe EDS"
    }


# ── Download ─────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, session: requests.Session,
             retries: int = MAX_RETRIES) -> tuple[bool, str]:
    """Resumable single-file download. Returns (success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, "exists"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=90, stream=True)
            if r.status_code == 404:
                return False, "404"
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
            tmp.rename(dest)
            return True, "downloaded"
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return False, str(e)
    return False, "max_retries"


def run_download(tasks: list, workers: int, label: str) -> tuple[int, int, int, list]:
    """Download a list of (pdb_id, url, dest) tasks on a thread pool.

    Returns (n_downloaded, n_exists, n_fail, failures[(pdb_id, msg)]).
    """
    n_dl = n_exist = n_fail = 0
    failures: list = []

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=MAX_RETRIES, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    with tqdm(total=len(tasks), unit="file", desc=label) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(download, url, dest, session): pid
                for pid, url, dest in tasks
            }
            for fut in as_completed(futs):
                pid = futs[fut]
                ok, msg = fut.result()
                if ok and msg == "exists":
                    n_exist += 1
                elif ok:
                    n_dl += 1
                else:
                    n_fail += 1
                    failures.append((pid, msg))
                pbar.set_postfix(dl=n_dl, have=n_exist, fail=n_fail, refresh=False)
                pbar.update(1)
    return n_dl, n_exist, n_fail, failures


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CCP4 maps + deposited PDBs for CrossDocked2020 density crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default=str(VOXDATA),
                   help="Directory containing data_train.pt / data_test.pt")
    p.add_argument("--ccp4_dir", default=str(VOXDATA / "ccp4"),
                   help="Output directory for {pdb}.ccp4 maps")
    p.add_argument("--pdb_dir", default=str(VOXDATA / "pdb"),
                   help="Output directory for {pdb}.pdb deposited coordinates")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "val", "test"],
                   help="Splits whose PDB IDs are downloaded")
    p.add_argument("--max_len", type=int, default=30,
                   help="Ligand size filter for PDB-ID discovery (match the dataset)")
    p.add_argument("--what", choices=["both", "maps", "pdbs"], default="both",
                   help="Which artifacts to download")
    p.add_argument("--eds_cache", default=None,
                   help="Optional EDS availability cache (check_xray_data.py) to "
                        "restrict MAP downloads to confirmed-EDS PDB IDs")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel download threads")
    p.add_argument("--dry_run", action="store_true",
                   help="Report counts without downloading")
    return p.parse_args()


def main():
    args = parse_args()
    ccp4_dir = Path(args.ccp4_dir)
    pdb_dir = Path(args.pdb_dir)
    ccp4_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = pdb_ids_from_splits(args.data_dir, args.splits, args.max_len)

    map_ids = pdb_ids
    if args.eds_cache:
        eds = load_eds_pdb_ids(Path(args.eds_cache))
        map_ids = [p for p in pdb_ids if p in eds]

    print("=== X-ray density download (CCP4 maps + deposited PDBs) ===")
    print(f"  data_dir   : {Path(args.data_dir).resolve()}")
    print(f"  splits     : {args.splits}  (max_len {args.max_len})")
    print(f"  unique PDBs: {len(pdb_ids):,}")
    if args.eds_cache:
        print(f"  EDS cache  : {args.eds_cache}  → {len(map_ids):,} maps with confirmed EDS")
    print(f"  what       : {args.what}")
    print(f"  ccp4_dir   : {ccp4_dir.resolve()}")
    print(f"  pdb_dir    : {pdb_dir.resolve()}")
    print(f"  workers    : {args.workers}")

    # Build task lists (download() skips existing files, but pre-filtering keeps
    # the progress bar honest about how much is actually left to fetch).
    map_tasks = [
        (pid, MAP_URL.format(pid=pid), ccp4_dir / f"{pid}.ccp4")
        for pid in map_ids
    ] if args.what in ("both", "maps") else []
    pdb_tasks = [
        (pid, PDB_URL.format(pid=pid), pdb_dir / f"{pid}.pdb")
        for pid in pdb_ids
    ] if args.what in ("both", "pdbs") else []

    def _pending(tasks):
        return sum(1 for _, _, d in tasks if not (d.exists() and d.stat().st_size > 0))

    print(f"\n  maps : {len(map_tasks):,} total, {_pending(map_tasks):,} to fetch")
    print(f"  pdbs : {len(pdb_tasks):,} total, {_pending(pdb_tasks):,} to fetch")

    if args.dry_run:
        print("\n[dry-run] nothing downloaded.")
        return

    if map_tasks:
        dl, ex, fa, fails = run_download(map_tasks, args.workers, "maps")
        print(f"\n── maps ── downloaded {dl:,} | already had {ex:,} | failed {fa:,}")
        if fails:
            log = ccp4_dir / "download_failures.txt"
            log.write_text("\n".join(f"{p}\t{m}" for p, m in fails))
            print(f"   failure log → {log}")

    if pdb_tasks:
        dl, ex, fa, fails = run_download(pdb_tasks, args.workers, "pdbs")
        print(f"\n── pdbs ── downloaded {dl:,} | already had {ex:,} | failed {fa:,}")
        if fails:
            log = pdb_dir / "download_failures.txt"
            log.write_text("\n".join(f"{p}\t{m}" for p, m in fails))
            print(f"   failure log → {log}")

    print("\nDone.  Next: dataset/00b_density_preprocess.py")


if __name__ == "__main__":
    main()
