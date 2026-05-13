"""04b_download_xray_mtz.py — Download 2Fo-Fc map coefficients from RCSB validation
reports and convert to CCP4 maps via gemmi.

Unlike 04_download_xray.py (PDBe EDS pre-computed CCP4), this pipeline fetches
the Fourier map coefficients (pdbx_FWT / pdbx_PHWT) from the RCSB PDB validation
CIF files and computes the real-space map on-the-fly with gemmi.  The result is
written to the same output directory and filename convention as the EDS pipeline
(dataset/data/ccp4/{pdb_id}.map), so 05_process_xray.py / 07_process_xray.sh
work unchanged.

Source URL
----------
    https://files.rcsb.org/pub/pdb/validation_reports/{mid2}/{pdb_id}/
        {pdb_id}_validation_2fo-fc_map_coef.cif.gz

Usage
-----
    cd voxbind
    python scripts/04b_download_xray_mtz.py \\
        --eds_cache ../notebook/data/check_xray_cache.json \\
        --out_dir   dataset/data/ccp4 \\
        --workers   8

    # Or provide an explicit list of PDB IDs:
    python scripts/04b_download_xray_mtz.py \\
        --pdb_ids 5p9s 1a2b 3d4e \\
        --out_dir dataset/data/ccp4

Outputs
-------
    dataset/data/ccp4/{pdb_id}.map   CCP4/MRC format, 2Fo-Fc map
    dataset/data/ccp4/mtz_failures.txt   PDB IDs that could not be fetched/converted
"""

import argparse
import gzip
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi is required.  pip install gemmi", file=sys.stderr)
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────────

RCSB_VALIDATION_URL = (
    "https://files.rcsb.org/pub/pdb/validation_reports"
    "/{mid2}/{pdb_id}/{pdb_id}_validation_2fo-fc_map_coef.cif.gz"
)
SAMPLE_RATE = 3.0   # grid oversampling for transform_f_phi_to_map (≥ 2.5 recommended)
RETRY_WAIT  = 2.0
MAX_RETRIES = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_pdb_ids_from_cache(eds_cache_path: Path) -> list[str]:
    """Return all X-ray PDB IDs from the EDS availability cache."""
    cache = json.loads(eds_cache_path.read_text())
    ids = [
        info["pdb_id"].lower()
        for info in cache.values()
        if info.get("has_density") and info.get("source") == "PDBe EDS"
    ]
    return sorted(set(ids))


def cif_gz_to_ccp4(cif_gz_bytes: bytes, out_path: Path) -> None:
    """Decompress CIF.gz bytes, convert map coefficients to CCP4, write to out_path."""
    cif_bytes = gzip.decompress(cif_gz_bytes)

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(cif_bytes)

    try:
        doc     = gemmi.cif.read(str(tmp_path))
        rblocks = gemmi.as_refln_blocks(doc)
        if not rblocks:
            raise ValueError("No reflection blocks in CIF")

        rb   = rblocks[0]
        grid = rb.transform_f_phi_to_map("pdbx_FWT", "pdbx_PHWT",
                                         sample_rate=SAMPLE_RATE)

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()

        tmp_out = out_path.with_suffix(".tmp")
        ccp4.write_ccp4_map(str(tmp_out))
        tmp_out.rename(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def download_and_convert(
    pdb_id: str,
    out_dir: Path,
    session: requests.Session,
) -> tuple[str, bool, str]:
    """Download 2Fo-Fc CIF.gz from RCSB and convert to CCP4.

    Returns (pdb_id, success, message).
    """
    out_path = out_dir / f"{pdb_id}.map"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pdb_id, True, "exists"

    url = RCSB_VALIDATION_URL.format(pdb_id=pdb_id, mid2=pdb_id[1:3])

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return pdb_id, False, "404 — no validation report on RCSB"
            r.raise_for_status()

            cif_gz_to_ccp4(r.content, out_path)
            return pdb_id, True, "downloaded+converted"

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return pdb_id, False, f"request error: {e}"
        except Exception as e:
            return pdb_id, False, f"conversion error: {e}"

    return pdb_id, False, "max retries exceeded"


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download RCSB validation 2Fo-Fc map coefficients and convert to CCP4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--eds_cache",
        help="Path to EDS availability cache (from check_xray_data.py). "
             "All X-ray PDB IDs in the cache are downloaded.",
    )
    src.add_argument(
        "--pdb_ids",
        nargs="+",
        metavar="PDB_ID",
        help="Explicit list of PDB IDs to download (overrides --eds_cache).",
    )
    p.add_argument(
        "--out_dir",
        default="dataset/data/ccp4",
        help="Output directory for .map files",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download/conversion threads",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without doing it",
    )
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDB IDs
    if args.pdb_ids:
        pdb_ids = [pid.lower() for pid in args.pdb_ids]
    else:
        cache_path = Path(args.eds_cache)
        if not cache_path.exists():
            print(f"ERROR: EDS cache not found: {cache_path}", file=sys.stderr)
            print("  Run notebook/check_xray_data.py first.", file=sys.stderr)
            sys.exit(1)
        pdb_ids = load_pdb_ids_from_cache(cache_path)

    already     = [pid for pid in pdb_ids
                   if (out_dir / f"{pid}.map").exists()
                   and (out_dir / f"{pid}.map").stat().st_size > 0]
    to_download = [pid for pid in pdb_ids if pid not in set(already)]

    print("=== RCSB validation map download + conversion ===")
    print(f"  Total PDB IDs   : {len(pdb_ids):,}")
    print(f"  Already present : {len(already):,}")
    print(f"  To download     : {len(to_download):,}")
    print(f"  Output dir      : {out_dir.resolve()}")
    print(f"  Workers         : {args.workers}")
    print(f"  Sample rate     : {SAMPLE_RATE}")

    if args.dry_run:
        print("\n[dry-run] First 20 URLs:")
        for pid in to_download[:20]:
            print(f"  {RCSB_VALIDATION_URL.format(pdb_id=pid, mid2=pid[1:3])}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download)-20} more")
        return

    if not to_download:
        print("\nAll maps already present. Nothing to do.")
        return

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=MAX_RETRIES, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)

    n_ok, n_fail = 0, 0
    failures: list[tuple[str, str]] = []

    with tqdm(total=len(to_download), unit="map", desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(download_and_convert, pid, out_dir, session): pid
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

    print(f"\n── Summary {'─'*35}")
    print(f"  Downloaded + converted : {n_ok:,}")
    print(f"  Failed                 : {n_fail:,}")
    if failures:
        fail_log = out_dir / "mtz_failures.txt"
        fail_log.write_text("\n".join(f"{p}\t{m}" for p, m in failures))
        print(f"  Failure log            : {fail_log}")
    print(f"  Maps saved             : {out_dir.resolve()}")
    print("─" * 46)


if __name__ == "__main__":
    main()
