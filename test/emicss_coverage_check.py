"""
EMICSS Coverage Check
======================
Compares two strategies for mapping CrossDocked PDB IDs → EMDB accessions:

  (A) RCSB / UniProt route (existing):   PDB → UniProt → EM-PDB structures → EMDB
  (B) EMICSS / PDBe route (this script): PDB → EMDB directly via fitted-model cross-reference
      Endpoint: https://www.ebi.ac.uk/emdb/api/search/{pdb_id}
      Returns EMDB entries for which {pdb_id} is a deposited fitted atomic model.

Strategy B gives higher *precision* (the EMDB map literally contains this PDB structure as
a fitted model), at the cost of lower *recall* (only works when the same PDB ID was used
as a fitted model, rare for X-ray CrossDocked structures).

This script does NOT download any maps — it only collects EMDB accessions.

Usage
-----
    conda run -n voxbind python test/emicss_coverage_check.py

All HTTP results are cached in test/data/emicss_pdb_to_emdb.json so the
script can be interrupted and resumed.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).parent.parent
DATA_DIR      = REPO_ROOT / "voxbind" / "dataset" / "data"
NB_DATA_DIR   = REPO_ROOT / "notebook" / "data"
OUT_DIR       = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RCSB_MAPPING  = NB_DATA_DIR / "crossdocked_pdb_to_emdb.json"   # existing strategy A
EMICSS_CACHE  = OUT_DIR / "emicss_pdb_to_emdb.json"            # strategy B (this script)

# ── API ───────────────────────────────────────────────────────────────────────
EMICSS_URL  = "https://www.ebi.ac.uk/emdb/api/search/{pdb_id}"
MAX_WORKERS = 16
SAVE_EVERY  = 200
RETRY_WAIT  = 2.0   # seconds between retries on 5xx


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_pdb_id(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].upper()


def fetch_emicss(pdb_id: str, session: requests.Session) -> tuple[str, list[str]]:
    """
    Query the EMDB fitted-model search for one PDB ID.

    Returns (pdb_id, [emdb_id, ...]).
    An empty list means no EMDB map lists this PDB as a fitted model.
    """
    url = EMICSS_URL.format(pdb_id=pdb_id.lower())
    for attempt in range(3):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                entries = r.json()
                if not isinstance(entries, list):
                    return pdb_id, []
                return pdb_id, [e["emdb_id"] for e in entries if "emdb_id" in e]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(RETRY_WAIT * (attempt + 1))
                continue
            return pdb_id, []          # 404 or other client error → no results
        except Exception:
            time.sleep(RETRY_WAIT)
    return pdb_id, []


# ── Stage 1: extract unique PDB IDs ──────────────────────────────────────────

def load_pdb_ids() -> list[str]:
    pdb_ids: set[str] = set()
    for fname in ("data_train.pt", "data_test.pt"):
        pt_path = DATA_DIR / fname
        if not pt_path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        samples = torch.load(pt_path, weights_only=False)
        for pocket_, _ in samples:
            pdb_ids.add(extract_pdb_id(pocket_["id"]))
        print(f"  {fname}: {len(samples):,} samples")
    return sorted(pdb_ids)


# ── Stage 2: EMICSS lookup ────────────────────────────────────────────────────

def fetch_all_emicss(pdb_ids: list[str]) -> dict[str, list[str]]:
    """Load cache, resolve missing entries, save and return."""
    if EMICSS_CACHE.exists():
        with open(EMICSS_CACHE) as f:
            cache: dict[str, list[str]] = json.load(f)
        print(f"  Loaded cache: {len(cache):,} entries")
    else:
        cache = {}

    to_resolve = [p for p in pdb_ids if p not in cache]
    if not to_resolve:
        print("  All cached.")
        return cache

    print(f"  Querying EMICSS for {len(to_resolve):,} PDB IDs …")
    session = requests.Session()
    n_done  = 0

    with tqdm(total=len(to_resolve), unit="PDB", desc="  EMICSS lookup") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_emicss, p, session): p for p in to_resolve}
            for fut in as_completed(futures):
                pid, emdb_ids = fut.result()
                cache[pid] = emdb_ids
                n_done += 1
                pbar.update(1)
                if n_done % SAVE_EVERY == 0:
                    with open(EMICSS_CACHE, "w") as f:
                        json.dump(cache, f)

    with open(EMICSS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Saved → {EMICSS_CACHE}")
    return cache


# ── Stage 3: compare and report ───────────────────────────────────────────────

def report(
    pdb_ids: list[str],
    emicss: dict[str, list[str]],
    rcsb: dict[str, list[str]],
) -> None:
    total = len(pdb_ids)

    # Coverage
    emicss_hits = {p for p in pdb_ids if emicss.get(p)}
    rcsb_hits   = {p for p in pdb_ids if rcsb.get(p)}
    both_hits   = emicss_hits & rcsb_hits
    only_emicss = emicss_hits - rcsb_hits
    only_rcsb   = rcsb_hits   - emicss_hits

    # Per-PDB EMDB count
    emicss_counts = [len(emicss.get(p, [])) for p in pdb_ids if emicss.get(p)]
    rcsb_counts   = [len(rcsb.get(p,   [])) for p in pdb_ids if rcsb.get(p)]

    import statistics
    def fmt_counts(counts):
        if not counts:
            return "n/a"
        return (f"median={statistics.median(counts):.0f}"
                f"  mean={statistics.mean(counts):.1f}"
                f"  max={max(counts)}")

    # Unique EMDB IDs
    emicss_all = {e for v in emicss.values() for e in v}
    rcsb_all   = {e for v in rcsb.values()   for e in v}

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  CrossDocked unique PDB IDs : {total:,}")
    print(sep)
    print(f"{'Strategy':<30s}  {'Coverage':>10s}  {'Unique EMDB IDs':>16s}")
    print(sep)
    print(f"{'(A) RCSB/UniProt':<30s}  "
          f"{len(rcsb_hits):>7,} ({100*len(rcsb_hits)/total:.1f}%)  "
          f"{len(rcsb_all):>16,}")
    print(f"{'(B) EMICSS/fitted-model':<30s}  "
          f"{len(emicss_hits):>7,} ({100*len(emicss_hits)/total:.1f}%)  "
          f"{len(emicss_all):>16,}")
    print(sep)
    print(f"  Both strategies match      : {len(both_hits):,}")
    print(f"  Only EMICSS (not in RCSB)  : {len(only_emicss):,}")
    print(f"  Only RCSB   (not EMICSS)   : {len(only_rcsb):,}")
    print(sep)
    print(f"  EMDB IDs per matched PDB:")
    print(f"    (A) RCSB/UniProt   : {fmt_counts(rcsb_counts)}")
    print(f"    (B) EMICSS         : {fmt_counts(emicss_counts)}")
    print(sep)

    # Breakdown of EMICSS relationship types (precision signal)
    print("\n  Sample EMICSS matches (first 10):")
    shown = 0
    for pid in sorted(emicss_hits):
        if shown >= 10:
            break
        ids = emicss.get(pid, [])
        rcsb_ids = rcsb.get(pid, [])
        overlap = set(ids) & set(rcsb_ids)
        print(f"    {pid:8s} → EMICSS: {ids}  | also in RCSB: {sorted(overlap)}")
        shown += 1

    # Save comparison table
    comparison = {}
    for pid in pdb_ids:
        e = emicss.get(pid, [])
        r2 = rcsb.get(pid, [])
        if e or r2:
            comparison[pid] = {
                "emicss": sorted(e),
                "rcsb":   sorted(r2),
                "overlap": sorted(set(e) & set(r2)),
            }
    out_path = OUT_DIR / "emicss_vs_rcsb_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Full comparison saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EMICSS vs RCSB/UniProt coverage check")
    print("=" * 60)

    # Load existing RCSB mapping (strategy A)
    if not RCSB_MAPPING.exists():
        sys.exit(f"[ERROR] RCSB mapping not found: {RCSB_MAPPING}\n"
                 "Run test/fetch_emdb_maps.py first.")
    with open(RCSB_MAPPING) as f:
        rcsb_mapping: dict[str, list[str]] = json.load(f)

    print("\n[Stage 1] Loading PDB IDs from CrossDocked .pt files …")
    pdb_ids = load_pdb_ids()
    print(f"  Unique PDB IDs: {len(pdb_ids):,}")

    print("\n[Stage 2] EMICSS fitted-model lookup …")
    emicss_mapping = fetch_all_emicss(pdb_ids)

    print("\n[Stage 3] Coverage comparison …")
    report(pdb_ids, emicss_mapping, rcsb_mapping)

    print("\nDone.")


if __name__ == "__main__":
    main()
