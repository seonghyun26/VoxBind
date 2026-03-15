"""04_download_xray.py — Download 2Fo-Fc CCP4 maps from PDBe EDS

Downloads X-ray electron density maps for all CrossDocked2020 PDB IDs that
have confirmed EDS structure factors (as determined by check_xray_data.py).

Usage
-----
    cd voxbind
    python scripts/04_download_xray.py \
        --eds_cache ../notebook/data/check_xray_cache.json \
        --out_dir   dataset/data/ccp4 \
        --workers   16

Outputs
-------
    dataset/data/ccp4/{pdb_id}.map   (CCP4 / MRC format, 2Fo-Fc map)

The download is resumable: already-existing files are skipped.
Only structures with source="PDBe EDS" are downloaded (X-ray; cryo-EM EMDB
maps are handled separately).

PDBe EDS download URL
---------------------
    https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}_2fofc.map
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

# PDBE_EDS_MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}_2fofc.map"
PDBE_EDS_MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4"
RETRY_WAIT = 2.0
MAX_RETRIES = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_eds_pdb_ids(eds_cache_path: Path) -> list[str]:
    """Return lowercase PDB IDs with confirmed PDBe EDS X-ray density."""
    if not eds_cache_path.exists():
        print(f"[ERROR] EDS cache not found: {eds_cache_path}", file=sys.stderr)
        print("  Run notebook/check_xray_data.py first.", file=sys.stderr)
        sys.exit(1)

    cache = json.loads(eds_cache_path.read_text())
    pdb_ids = [
        info["pdb_id"].lower()
        for info in cache.values()
        if info.get("has_density") and info.get("source") == "PDBe EDS"
    ]
    return sorted(set(pdb_ids))


def download_map(pdb_id: str, out_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    """Download 2Fo-Fc CCP4 map for one PDB ID.

    Returns (pdb_id, success, message).
    """
    out_path = out_dir / f"{pdb_id}.map"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pdb_id, True, "exists"

    url = PDBE_EDS_MAP_URL.format(pdb_id=pdb_id)
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=60, stream=True)
            if r.status_code == 404:
                return pdb_id, False, "404 not found"
            r.raise_for_status()

            # Write in chunks to avoid large memory footprint
            tmp = out_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
            tmp.rename(out_path)
            return pdb_id, True, "downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return pdb_id, False, str(e)

    return pdb_id, False, "max retries exceeded"


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PDBe EDS 2Fo-Fc maps for CrossDocked2020",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--eds_cache",
        default="../notebook/data/check_xray_cache.json",
        help="Path to EDS availability cache from check_xray_data.py",
    )
    p.add_argument(
        "--out_dir",
        default="dataset/data/ccp4",
        help="Directory to store downloaded .map files",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download threads",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    return p.parse_args()


def main():
    args = parse_args()

    eds_cache = Path(args.eds_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load confirmed PDB IDs
    pdb_ids = load_eds_pdb_ids(eds_cache)
    already = sum(1 for pid in pdb_ids if (out_dir / f"{pid}.map").exists()
                  and (out_dir / f"{pid}.map").stat().st_size > 0)
    to_download = [pid for pid in pdb_ids
                   if not (out_dir / f"{pid}.map").exists()
                   or (out_dir / f"{pid}.map").stat().st_size == 0]

    print(f"=== PDBe EDS CCP4 map download ===")
    print(f"  Total PDB IDs with EDS : {len(pdb_ids):,}")
    print(f"  Already downloaded     : {already:,}")
    print(f"  To download            : {len(to_download):,}")
    print(f"  Output directory       : {out_dir.resolve()}")
    print(f"  Workers                : {args.workers}")

    if args.dry_run:
        print("\n[dry-run] Would download:")
        for pid in to_download[:20]:
            print(f"  {PDBE_EDS_MAP_URL.format(pdb_id=pid)}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        return

    if not to_download:
        print("\nAll maps already present. Nothing to do.")
        return

    n_ok = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=MAX_RETRIES, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504]
        )
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    with tqdm(total=len(to_download), unit="map", desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(download_map, pid, out_dir, session): pid
                for pid in to_download
            }
            for fut in as_completed(futs):
                pdb_id, ok, msg = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append((pdb_id, msg))
                pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
                pbar.update(1)

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Downloaded  : {n_ok:,}")
    print(f"  Failed      : {n_fail:,}")
    if failures:
        fail_log = out_dir / "download_failures.txt"
        fail_log.write_text("\n".join(f"{p}\t{m}" for p, m in failures))
        print(f"  Failure log : {fail_log}")
    print(f"  Maps saved  : {out_dir.resolve()}")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    main()
