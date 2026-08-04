"""pocket.py — cache binding-pocket residue sequences for the v_pkt branch.

Parses each complex's {pdb}_pocket.pdb (Biopython) into a one-letter residue
sequence, cached once to cache/pocket_seqs.json so training doesn't re-parse PDBs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, complex_files
from featurize import pocket_sequence_from_pdb
from manifest import build_manifest

POCKET_CACHE = CACHE_DIR / "pocket_seqs.json"


def load_pocket_cache() -> dict[str, str]:
    if POCKET_CACHE.exists():
        return json.loads(POCKET_CACHE.read_text())
    return {}


def build_pocket_cache(manifest, overwrite: bool = False) -> dict[str, str]:
    cache = {} if overwrite else load_pocket_cache()
    n_new = n_empty = 0
    for k, (_, r) in enumerate(manifest.iterrows()):
        if r.pdb_id in cache and not overwrite:
            continue
        pkt = complex_files(r.pdb_id)["pocket"]
        seq = pocket_sequence_from_pdb(str(pkt)) if pkt.exists() else ""
        cache[r.pdb_id] = seq
        n_new += 1
        n_empty += int(len(seq) == 0)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(manifest)}  new={n_new} empty={n_empty}", flush=True)
    POCKET_CACHE.write_text(json.dumps(cache))
    lens = [len(v) for v in cache.values() if v]
    print(f"DONE  cached={len(cache)}  new={n_new}  empty={n_empty}  "
          f"pocket_len min/median/max="
          f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}", flush=True)
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build pocket-sequence cache")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cl1_only", action="store_true")
    ap.add_argument("--keep_covalent", action="store_true")
    args = ap.parse_args()
    m = build_manifest(cl1_only=args.cl1_only, drop_covalent=not args.keep_covalent)
    build_pocket_cache(m, overwrite=args.overwrite)
