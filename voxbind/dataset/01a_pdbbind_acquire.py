"""01a_pdbbind_acquire.py — Acquire PDBbind v2020 (structures, index, density).

Consolidated Phase 1-3 entry point. Three subcommands, run in pipeline order:

    cd voxbind
    python dataset/01a_pdbbind_acquire.py structures   # download zip + LP-PDBbind CSV
    python dataset/01a_pdbbind_acquire.py index         # extract zip + build index.csv
    python dataset/01a_pdbbind_acquire.py density        # 2Fo-Fc CCP4 maps from PDBe EDS

──────────────────────────────────────────────────────────────────────────────
structures — Download PDBbind v2020 refined set + LP-PDBbind metadata
──────────────────────────────────────────────────────────────────────────────
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

    python dataset/01a_pdbbind_acquire.py structures             # download both
    python dataset/01a_pdbbind_acquire.py structures --verify    # verify only
    python dataset/01a_pdbbind_acquire.py structures --no_hash   # skip sha256
  → dataset/data/pdbbind/raw/{pbpp-2020.zip, LP_PDBBind.csv, .sha256.txt}

──────────────────────────────────────────────────────────────────────────────
index — Extract PDBbind v2020 refined set + build canonical index
──────────────────────────────────────────────────────────────────────────────
Extracts the zip into dataset/data/pdbbind/structures/<pdb_id>/{protein,pocket,
ligand}.* and writes a single canonical index file:
  - dataset/data/pdbbind/index.csv
    columns: pdb_id, set, pK, new_split, CL1, CL2, CL3, covalent, has_struct

    python dataset/01a_pdbbind_acquire.py index
    python dataset/01a_pdbbind_acquire.py index --force_extract   # re-extract zip
  → structures/<pdb_id>/...   index.csv   missing.txt

──────────────────────────────────────────────────────────────────────────────
density — 2Fo-Fc CCP4 maps from PDBe EDS
──────────────────────────────────────────────────────────────────────────────
For each PDBbind v2020 refined complex on disk (has_struct=True in index.csv):
  1. HEAD-probe PDBe EDS to see if a 2Fo-Fc map exists for this PDB ID.
  2. Cache the result so re-runs don't re-probe.
  3. Download the map for every pdb_id that has one.

Maps are stored uncropped — the per-ligand-COM crop happens in preprocessing
(01b_pdbbind_preprocess.py voxelize). Beyond Atoms (§3.1) centres ligand+pocket
grids on ligand COM at 0.25 Å resolution.

    python dataset/01a_pdbbind_acquire.py density
    python dataset/01a_pdbbind_acquire.py density --check_only   # stop after HEAD
    python dataset/01a_pdbbind_acquire.py density --workers 32
  → eds_cache.json   ccp4/{pdb_id}.ccp4   ccp4/download_failures.txt

    PDBe EDS endpoint: https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4
"""

import argparse
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
RAW_DIR     = PDBBIND_DIR / "raw"
STRUCT_DIR  = PDBBIND_DIR / "structures"
INDEX_CSV   = PDBBIND_DIR / "index.csv"
EDS_CACHE   = PDBBIND_DIR / "eds_cache.json"
CCP4_DIR    = PDBBIND_DIR / "ccp4"

ZIP_NAME = "pbpp-2020.zip"
CSV_NAME = "LP_PDBBind.csv"

# Per-complex file naming pattern (substituted with pdb_id at check time)
REQUIRED_PER_PDBID = [
    "{pdb_id}_protein.pdb",
    "{pdb_id}_pocket.pdb",
    "{pdb_id}_ligand.mol2",
    "{pdb_id}_ligand.sdf",
]

# structures — HuggingFace + GitHub bulk downloads.
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
STRUCT_MAX_RETRIES   = 4
STRUCT_RETRY_BACKOFF = 2.0
CHUNK = 1 << 20  # 1 MB

# density — PDBe EDS per-map downloads.
PDBE_EDS_MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4"
EDS_RETRY_WAIT  = 2.0
EDS_MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# structures — download PDBbind v2020 refined set + LP-PDBbind metadata
# ═══════════════════════════════════════════════════════════════════════════════

def download(url: str, dst: Path, session: requests.Session) -> bool:
    """Stream `url` to `dst`. Supports HTTP Range resume on partial files."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    headers: dict[str, str] = {}
    resume_from = 0

    if tmp.exists():
        resume_from = tmp.stat().st_size
        headers["Range"] = f"bytes={resume_from}-"

    for attempt in range(STRUCT_MAX_RETRIES):
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
            print(f"  [retry {attempt+1}/{STRUCT_MAX_RETRIES}] {e}")
            if attempt < STRUCT_MAX_RETRIES - 1:
                time.sleep(STRUCT_RETRY_BACKOFF * (attempt + 1))
                if tmp.exists():
                    resume_from = tmp.stat().st_size
                    headers["Range"] = f"bytes={resume_from}-"
            else:
                return False
    return False


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


def run_structures(args: argparse.Namespace) -> None:
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
        print("Next:  python dataset/01a_pdbbind_acquire.py index")
        sys.exit(0)
    else:
        print("One or more files missing or wrong size.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# index — extract PDBbind v2020 refined set + build index.csv
# ═══════════════════════════════════════════════════════════════════════════════

def is_populated(d: Path) -> bool:
    if not d.exists():
        return False
    return any(d.iterdir())


def extract_zip(zip_path: Path, dst_root: Path, force: bool) -> None:
    """Extract pbpp-2020.zip into dst_root/<pdb_id>/... .

    The zip's internal layout is determined at extract time — we just unpack
    everything and then normalize per-complex paths if needed.
    """
    if is_populated(dst_root) and not force:
        n_dirs = sum(1 for p in dst_root.iterdir() if p.is_dir())
        print(f"  [skip] {dst_root}/  ({n_dirs} entries, --force_extract to redo)")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {zip_path.name}  →  {dst_root}/")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=zip_path.name, unit="file"):
            zf.extract(m, dst_root)


def detect_struct_root(struct_root: Path) -> Path:
    """Find the directory that actually holds <pdb_id>/ children.

    The zip may extract to struct_root/ directly, or into one of several
    sibling subdirs (e.g. pbpp-2020/ plus readme/). Pick whichever path has
    the most 4-char alnum children.
    """
    if not struct_root.exists():
        return struct_root

    def n_pdbids(d: Path) -> int:
        try:
            return sum(1 for p in d.iterdir()
                       if p.is_dir() and len(p.name) == 4 and p.name.isalnum())
        except OSError:
            return 0

    best = struct_root
    best_count = n_pdbids(struct_root)
    for sub in struct_root.iterdir():
        if sub.is_dir():
            c = n_pdbids(sub)
            if c > best_count:
                best = sub
                best_count = c
    return best


def load_lp_pdbbind(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # First column is the pdb_id (saved without header → "Unnamed: 0").
    # Also accept "pdbid" if a newer version of the CSV uses that.
    pdb_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else "pdbid"
    df = df.rename(columns={pdb_col: "pdb_id"})

    # pK column — LP-PDBbind uses "value"; tolerate variants from other releases.
    pK_col = next((c for c in ["value", "-logKd/Ki", "pK", "pK_value"]
                   if c in df.columns), None)
    if pK_col is None:
        raise KeyError(f"Could not locate pK column in {csv_path}; "
                       f"saw columns: {list(df.columns)}")
    df = df.rename(columns={pK_col: "pK"})

    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def attach_struct_membership(df: pd.DataFrame, struct_root: Path) -> pd.DataFrame:
    if not struct_root.exists():
        df["has_struct"] = False
        return df

    def has(pid: str) -> bool:
        d = struct_root / pid
        if not d.exists():
            return False
        return all((d / fn.format(pdb_id=pid)).exists() for fn in REQUIRED_PER_PDBID)

    df["has_struct"] = df["pdb_id"].apply(has)
    return df


def run_index(args: argparse.Namespace) -> None:
    raw_dir    = Path(args.raw_dir).resolve()
    struct_dir = Path(args.struct_dir).resolve()
    index_out  = Path(args.index_out).resolve()

    print("=== PDBbind v2020 index build ===")
    print(f"  raw_dir    : {raw_dir}")
    print(f"  struct_dir : {struct_dir}")
    print(f"  index_out  : {index_out}")

    # Sanity: raw downloads present?
    zip_path = raw_dir / ZIP_NAME
    csv_path = raw_dir / CSV_NAME
    missing = [p for p in (zip_path, csv_path) if not p.exists()]
    if missing:
        print("\n[error] missing input(s):")
        for p in missing:
            print(f"    - {p}")
        print("Run: python dataset/01a_pdbbind_acquire.py structures")
        sys.exit(1)

    # Extract
    print("\n── Extract ─────────────────────────────────────────────────────")
    extract_zip(zip_path, struct_dir, force=args.force_extract)

    # Locate the actual <pdb_id>/ root inside what we extracted
    real_root = detect_struct_root(struct_dir)
    if real_root != struct_dir:
        print(f"  [info] real structure root is {real_root.relative_to(struct_dir.parent)}")

    # Load LP-PDBbind metadata
    print("\n── Load LP_PDBBind.csv ─────────────────────────────────────────")
    df = load_lp_pdbbind(csv_path)
    print(f"  {len(df):,} rows; columns: {list(df.columns)}")

    # Attach structural membership
    df = attach_struct_membership(df, real_root)

    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  total rows         : {len(df):,}")
    print(f"  has_struct (True)  : {df['has_struct'].sum():,}")
    print(f"  has_struct (False) : {(~df['has_struct']).sum():,}")
    if "category" in df.columns:
        print("\n  category  ×  has_struct:")
        print(df.groupby(["category", "has_struct"]).size().unstack(fill_value=0).to_string())
    if "new_split" in df.columns:
        print("\n  new_split  ×  has_struct:")
        print(df.groupby(["new_split", "has_struct"], dropna=False).size().unstack(fill_value=0).to_string())

    # Write missing list (entries in CSV but no structure on disk; mostly general-set)
    miss_path = PDBBIND_DIR / "missing.txt"
    miss_ids = df.loc[~df["has_struct"], "pdb_id"].tolist()
    if miss_ids:
        miss_path.write_text("\n".join(miss_ids) + "\n")
        print(f"\n  {len(miss_ids):,} pdb_ids not on disk  →  {miss_path}")

    # Write final index — keep useful columns, in stable order
    keep_cols = [c for c in
                 ["pdb_id", "has_struct", "category", "pK", "new_split",
                  "CL1", "CL2", "CL3", "covalent",
                  "resolution", "date", "type", "kd/ki"]
                 if c in df.columns]
    df_out = df[keep_cols].copy()
    df_out.to_csv(index_out, index=False)
    print(f"\n[write] {index_out}  ({len(df_out):,} rows, {len(keep_cols)} cols)")

    print("\nNext: python dataset/01a_pdbbind_acquire.py density")


# ═══════════════════════════════════════════════════════════════════════════════
# density — 2Fo-Fc CCP4 maps from PDBe EDS
# ═══════════════════════════════════════════════════════════════════════════════

def load_pdb_ids(index_csv: Path) -> list[str]:
    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)]
    return sorted(df["pdb_id"].str.lower().unique())


def make_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=EDS_MAX_RETRIES, backoff_factor=1.0,
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
    for attempt in range(EDS_MAX_RETRIES):
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
            if attempt < EDS_MAX_RETRIES - 1:
                time.sleep(EDS_RETRY_WAIT * (attempt + 1))
            else:
                return pdb_id, False, str(e)
    return pdb_id, False, "max retries exceeded"


def run_density(args: argparse.Namespace) -> None:
    index_csv = Path(args.index_csv)
    eds_cache = Path(args.eds_cache)
    ccp4_dir  = Path(args.ccp4_dir)
    ccp4_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        print(f"[error] missing {index_csv} — run: python dataset/01a_pdbbind_acquire.py index")
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Acquire PDBbind v2020 (structures | index | density)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser(
        "structures",
        help="Download PDBbind v2020 refined set + LP-PDBbind metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ps.add_argument("--raw_dir", default=str(RAW_DIR),
                    help="Directory to store downloaded archives")
    ps.add_argument("--verify", action="store_true",
                    help="Skip download, only verify what's already on disk")
    ps.add_argument("--no_hash", action="store_true",
                    help="Skip sha256 computation (faster on slow disks)")
    ps.set_defaults(func=run_structures)

    pi = sub.add_parser(
        "index",
        help="Extract pbpp-2020.zip and build index.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pi.add_argument("--raw_dir",    default=str(RAW_DIR))
    pi.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pi.add_argument("--index_out",  default=str(INDEX_CSV))
    pi.add_argument("--force_extract", action="store_true",
                    help="Re-extract pbpp-2020.zip even if structures/ is populated")
    pi.set_defaults(func=run_index)

    pden = sub.add_parser(
        "density",
        help="Download PDBe EDS 2Fo-Fc maps for the refined set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pden.add_argument("--index_csv",   default=str(INDEX_CSV))
    pden.add_argument("--eds_cache",   default=str(EDS_CACHE))
    pden.add_argument("--ccp4_dir",    default=str(CCP4_DIR))
    pden.add_argument("--workers",     type=int, default=16,
                      help="Parallel HEAD / GET threads")
    pden.add_argument("--check_only",  action="store_true",
                      help="Stop after availability check (no downloads)")
    pden.add_argument("--recheck",     action="store_true",
                      help="Recheck every pdb_id (ignore cached availability)")
    pden.set_defaults(func=run_density)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
