"""01a_pdbbind_acquire.py — Download PDBbind v2020 refined set + LP-PDBbind metadata.

PDBbind v2021 is paywalled on pdbbind-plus.org.cn, so we use v2020 — the last
freely-available release, and the same version Li et al. 2024 (LP-PDBbind, cited
by Beyond Atoms as the leakage fix) build on.

Sources (both public, no auth required):
  - HuggingFace  photonmz/pdbbindpp-2020       — PDBbind v2020 refined set
        5,316 complexes, ~600 MB compressed (~2.7 GB extracted)
        Each pdb_id/ has: protein.pdb, pocket.pdb, ligand.mol2, ligand.sdf
        'pp' = lightly prepared (heavy atoms preserved vs raw PDBbind)
  - GitHub       THGLab/LP-PDBBind/dataset     — LP_PDBBind.csv (~11 MB)
        Leak-proof train/val/test split + pK + clean-level flags

Usage
-----
    cd voxbind
    python dataset/01a_pdbbind_acquire.py             # download both
    python dataset/01a_pdbbind_acquire.py --verify    # verify only (no download)
    python dataset/01a_pdbbind_acquire.py --no_hash   # skip sha256

Outputs
-------
    dataset/data/pdbbind/raw/pbpp-2020.zip
    dataset/data/pdbbind/raw/LP_PDBBind.csv
    dataset/data/pdbbind/raw/.sha256.txt

Next
----
    python dataset/01b_pdbbind_index.py    # extract + build index
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent / "data" / "pdbbind" / "raw"

# (filename, url, (size_lo, size_hi))   sizes in bytes, ~±20%
DOWNLOADS: list[tuple[str, str, tuple[int, int]]] = [
    (
        "pbpp-2020.zip",
        "https://huggingface.co/datasets/photonmz/pdbbindpp-2020/resolve/main/pbpp-2020.zip",
        (500_000_000, 800_000_000),  # ~600 MB compressed (~2.7 GB extracted)
    ),
    (
        "LP_PDBBind.csv",
        "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv",
        (5_000_000, 20_000_000),  # ~11 MB
    ),
]

MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
CHUNK = 1 << 20  # 1 MB


# ── Download ───────────────────────────────────────────────────────────────────

def download(url: str, dst: Path, session: requests.Session) -> bool:
    """Stream `url` to `dst`. Supports HTTP Range resume on partial files."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    headers: dict[str, str] = {}
    resume_from = 0

    if tmp.exists():
        resume_from = tmp.stat().st_size
        headers["Range"] = f"bytes={resume_from}-"

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=headers, stream=True, timeout=120,
                            allow_redirects=True)
            if r.status_code == 416:
                # Range Not Satisfiable -> file already fully downloaded
                tmp.rename(dst)
                return True
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0)) + resume_from
            mode = "ab" if resume_from > 0 else "wb"

            with tmp.open(mode) as f, tqdm(
                total=total if total > 0 else None,
                initial=resume_from,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=dst.name, leave=True,
            ) as bar:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))

            tmp.rename(dst)
            return True

        except requests.exceptions.RequestException as e:
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                if tmp.exists():
                    resume_from = tmp.stat().st_size
                    headers["Range"] = f"bytes={resume_from}-"
            else:
                return False
    return False


# ── Verify ─────────────────────────────────────────────────────────────────────

def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        with tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                  unit_divisor=1024, desc=f"sha256 {path.name}", leave=False) as bar:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
                bar.update(len(block))
    return h.hexdigest()


def verify(raw_dir: Path, compute_hash: bool) -> tuple[bool, dict[str, str]]:
    print(f"=== Verifying {raw_dir} ===")
    ok = True
    hashes: dict[str, str] = {}
    for name, _url, (lo, hi) in DOWNLOADS:
        p = raw_dir / name
        if not p.exists():
            print(f"  [MISSING] {name}")
            ok = False
            continue
        size = p.stat().st_size
        size_ok = lo <= size <= hi
        marker = "OK     " if size_ok else "BAD-SZ "
        print(f"  [{marker}] {name}   ({size/1e9:.3f} GB)")
        if not size_ok:
            print(f"             expected size in [{lo/1e9:.2f}, {hi/1e9:.2f}] GB")
            ok = False
            continue
        if compute_hash:
            sha = sha256_of(p)
            hashes[name] = sha
            print(f"             sha256: {sha}")
    return ok, hashes


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PDBbind v2020 refined set + LP-PDBbind metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_dir", default=str(RAW_DIR),
                   help="Directory to store downloaded archives")
    p.add_argument("--verify", action="store_true",
                   help="Skip download, only verify what's already on disk")
    p.add_argument("--no_hash", action="store_true",
                   help="Skip sha256 computation (faster on slow disks)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=== PDBbind v2020 acquisition ===")
    print(f"  raw_dir   : {raw_dir}")
    print(f"  verify    : {args.verify}")
    print(f"  no_hash   : {args.no_hash}")

    if not args.verify:
        print("\n── Download ─────────────────────────────────────────────────────")
        session = requests.Session()
        for name, url, _ in DOWNLOADS:
            dst = raw_dir / name
            if dst.exists():
                print(f"  [skip] {name}  (already present at {dst.stat().st_size/1e9:.2f} GB)")
                continue
            print(f"  [GET]  {url}")
            ok = download(url, dst, session)
            if not ok:
                print(f"  [ERROR] failed to download {name}")
                sys.exit(2)

    print()
    ok, hashes = verify(raw_dir, compute_hash=not args.no_hash)

    if hashes:
        record = raw_dir / ".sha256.txt"
        record.write_text(
            "\n".join(f"{sha}  {name}" for name, sha in hashes.items()) + "\n"
        )
        print(f"\n  sha256 record → {record}")

    print()
    print("─" * 64)
    if ok:
        print(f"All {len(DOWNLOADS)}/{len(DOWNLOADS)} archives present.")
        print("Next:  python dataset/01b_pdbbind_index.py")
        sys.exit(0)
    else:
        print("One or more files missing or wrong size.")
        sys.exit(1)


if __name__ == "__main__":
    main()
