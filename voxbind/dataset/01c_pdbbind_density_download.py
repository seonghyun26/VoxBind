"""01c_pdbbind_density_download.py — Phase 3: 2Fo-Fc CCP4 maps from PDBe EDS.

For each PDBbind v2020 refined complex on disk (has_struct=True in index.csv):
  1. HEAD-probe PDBe EDS to see if a 2Fo-Fc map exists for this PDB ID.
  2. Cache the result so re-runs don't re-probe.
  3. Download the map for every pdb_id that has one.

This mirrors dataset/00a_data_density_download.py used for CrossDocked, but the
pdb_id list comes from dataset/data/pdbbind/index.csv instead of an EDS cache.

Maps are stored uncropped — the per-ligand-COM crop happens in Phase 4
(voxelization). Beyond Atoms (§3.1) centres ligand+pocket grids on ligand COM
at 0.25 Å resolution, so we'll defer the crop until the encoder spec is locked.

Usage
-----
    cd voxbind
    python dataset/01c_pdbbind_density_download.py
    python dataset/01c_pdbbind_density_download.py --check_only   # stop after HEAD
    python dataset/01c_pdbbind_density_download.py --workers 32

Outputs
-------
    dataset/data/pdbbind/eds_cache.json         per-pdb_id availability
    dataset/data/pdbbind/ccp4/{pdb_id}.ccp4     2Fo-Fc maps (only those with EDS)
    dataset/data/pdbbind/ccp4/download_failures.txt   (if any)

PDBe EDS endpoint
-----------------
    https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
INDEX_CSV   = PDBBIND_DIR / "index.csv"
EDS_CACHE   = PDBBIND_DIR / "eds_cache.json"
CCP4_DIR    = PDBBIND_DIR / "ccp4"

PDBE_EDS_MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4"
RETRY_WAIT  = 2.0
MAX_RETRIES = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_pdb_ids(index_csv: Path) -> list[str]:
    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)]
    return sorted(df["pdb_id"].str.lower().unique())


def make_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=MAX_RETRIES, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def check_eds(pdb_id: str, session: requests.Session) -> dict:
    """HEAD-probe PDBe EDS; return availability + Content-Length."""
    url = PDBE_EDS_MAP_URL.format(pdb_id=pdb_id)
    try:
        r = session.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            cl = r.headers.get("Content-Length")
            return {"pdb_id": pdb_id, "has_density": True,
                    "source": "PDBe EDS",
                    "content_length": int(cl) if cl else None}
        if r.status_code == 404:
            return {"pdb_id": pdb_id, "has_density": False,
                    "source": None, "content_length": None}
        return {"pdb_id": pdb_id, "has_density": False,
                "source": None, "content_length": None,
                "note": f"HTTP {r.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"pdb_id": pdb_id, "has_density": False,
                "source": None, "content_length": None,
                "note": str(e)[:120]}


def download_map(pdb_id: str, ccp4_dir: Path,
                 session: requests.Session) -> tuple[str, bool, str]:
    """Stream the 2Fo-Fc CCP4 map for `pdb_id` to disk; resumable via existence."""
    out_path = ccp4_dir / f"{pdb_id}.ccp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pdb_id, True, "exists"

    url = PDBE_EDS_MAP_URL.format(pdb_id=pdb_id)
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=120, stream=True)
            if r.status_code == 404:
                return pdb_id, False, "404 not found"
            r.raise_for_status()

            tmp = out_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
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
        description="Download PDBe EDS 2Fo-Fc maps for PDBbind v2020 refined set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index_csv",   default=str(INDEX_CSV))
    p.add_argument("--eds_cache",   default=str(EDS_CACHE))
    p.add_argument("--ccp4_dir",    default=str(CCP4_DIR))
    p.add_argument("--workers",     type=int, default=16,
                   help="Parallel HEAD / GET threads")
    p.add_argument("--check_only",  action="store_true",
                   help="Stop after availability check (no downloads)")
    p.add_argument("--recheck",     action="store_true",
                   help="Recheck every pdb_id (ignore cached availability)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    index_csv = Path(args.index_csv)
    eds_cache = Path(args.eds_cache)
    ccp4_dir  = Path(args.ccp4_dir)
    ccp4_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        print(f"[error] missing {index_csv} — run 01b_pdbbind_index.py first")
        sys.exit(1)

    pdb_ids = load_pdb_ids(index_csv)
    print("=== PDBbind v2020 refined — EDS density acquisition ===")
    print(f"  index_csv : {index_csv}")
    print(f"  eds_cache : {eds_cache}")
    print(f"  ccp4_dir  : {ccp4_dir}")
    print(f"  pdb_ids   : {len(pdb_ids):,}")
    print(f"  workers   : {args.workers}")

    # ── 1) Availability check (HEAD) ───────────────────────────────────────────
    cache: dict[str, dict] = {}
    if eds_cache.exists() and not args.recheck:
        cache = json.loads(eds_cache.read_text())
        print(f"\n[cache] loaded {len(cache):,} prior entries from {eds_cache.name}")

    to_check = [pid for pid in pdb_ids if pid not in cache]
    if to_check:
        print(f"\n── HEAD probes ─────────────────────────────────────────────────")
        print(f"  probing {len(to_check):,} pdb_ids ...")
        session = make_session()
        with tqdm(total=len(to_check), unit="id", desc="EDS check") as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(check_eds, pid, session): pid for pid in to_check}
                for fut in as_completed(futs):
                    info = fut.result()
                    cache[info["pdb_id"]] = info
                    bar.update(1)
        eds_cache.write_text(json.dumps(cache, indent=2))
        print(f"  [write] {eds_cache}")
    else:
        print("\n  [skip] all pdb_ids already cached")

    n_have    = sum(1 for v in cache.values() if v.get("has_density"))
    n_dont    = sum(1 for v in cache.values() if not v.get("has_density"))
    est_bytes = sum((v.get("content_length") or 0)
                    for v in cache.values() if v.get("has_density"))
    print(f"\n  with EDS density : {n_have:,} / {len(pdb_ids):,}  "
          f"({100*n_have/max(len(pdb_ids),1):.1f}%)")
    print(f"  without EDS      : {n_dont:,}")
    print(f"  est. total size  : {est_bytes/1e9:.2f} GB")

    if args.check_only:
        print("\n[--check_only] stopping after availability check.")
        return

    # ── 2) Download available maps ─────────────────────────────────────────────
    to_dl = [pid for pid, info in cache.items()
             if info.get("has_density")
             and not (ccp4_dir / f"{pid}.ccp4").exists()]
    n_already = sum(1 for pid, info in cache.items()
                    if info.get("has_density")
                    and (ccp4_dir / f"{pid}.ccp4").exists())

    print(f"\n── Download ────────────────────────────────────────────────────")
    print(f"  already on disk : {n_already:,}")
    print(f"  to fetch        : {len(to_dl):,}")

    if not to_dl:
        print("  Nothing to download.")
        return

    n_ok = n_fail = 0
    failures: list[tuple[str, str]] = []

    session = make_session()
    with tqdm(total=len(to_dl), unit="map", desc="Downloading") as bar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(download_map, pid, ccp4_dir, session): pid
                    for pid in to_dl}
            for fut in as_completed(futs):
                pid, ok, msg = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append((pid, msg))
                bar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
                bar.update(1)

    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Downloaded   : {n_ok:,}")
    print(f"  Failed       : {n_fail:,}")
    if failures:
        fail_log = ccp4_dir / "download_failures.txt"
        fail_log.write_text("\n".join(f"{p}\t{m}" for p, m in failures) + "\n")
        print(f"  Failure log  : {fail_log}")
    n_on_disk = sum(1 for p in ccp4_dir.iterdir() if p.suffix == ".ccp4")
    total_gb  = sum(p.stat().st_size for p in ccp4_dir.iterdir()
                    if p.suffix == ".ccp4") / 1e9
    print(f"  Maps on disk : {n_on_disk:,}  ({total_gb:.2f} GB)")
    print(f"  Maps saved   : {ccp4_dir}")


if __name__ == "__main__":
    main()
