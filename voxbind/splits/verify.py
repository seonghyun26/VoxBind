"""verify.py — reproducibility gate for the frozen split manifests.

Network-free, fast, runnable on every server before a train/probe run. For each
scheme it re-reads the committed ``pid,split`` CSV, recomputes the content hash,
and asserts it matches the pinned ``sha256`` + per-split counts in MANIFEST.json.
A corrupted or accidentally-regenerated manifest fails loudly (exit 1) instead of
silently shifting the split.

    cd /home/shpark/prj-denovo/VoxBind
    python voxbind/splits/verify.py            # all schemes; exit 1 on drift
    python voxbind/splits/verify.py --scheme lp_edrscc_v1

This checks the *integrity of the committed artifact* (is everyone using the same
frozen list?). It does NOT re-derive from primary inputs — that's make_splits.py's
job and needs the volatile data/ artifacts; this gate deliberately needs none.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from voxbind.splits import (  # noqa: E402
    SPLIT_NAMES,
    SPLITS_DIR,
    _read_manifest_csv,
    manifest_hash,
    read_manifest,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assert frozen split manifests are intact + reproducible.")
    ap.add_argument("--scheme", default="all")
    args = ap.parse_args(argv)

    manifest = read_manifest()
    schemes = manifest.get("schemes", {})
    want = sorted(schemes) if args.scheme == "all" else [args.scheme]

    print("=== frozen split manifests — integrity check ===")
    print(f"  manifest: {(SPLITS_DIR / 'MANIFEST.json').relative_to(REPO_ROOT)}\n")

    ok = True
    for scheme in want:
        if scheme not in schemes:
            print(f"  [FAIL] {scheme}: not in MANIFEST.json (known: {sorted(schemes)})")
            ok = False
            continue
        info = schemes[scheme]
        csv_path = SPLITS_DIR / info["file"]
        if not csv_path.exists():
            print(f"  [FAIL] {scheme}: manifest CSV missing → {csv_path}")
            ok = False
            continue
        pairs = _read_manifest_csv(csv_path)
        counts = {s: sum(1 for _, sp in pairs if sp == s) for s in SPLIT_NAMES}
        got_hash = manifest_hash(pairs)

        bad = []
        for s in SPLIT_NAMES:
            exp = int(info["counts"].get(s, -1))
            if exp >= 0 and counts[s] != exp:
                bad.append(f"{s} {counts[s]}!={exp}")
        if got_hash != info["sha256"]:
            bad.append("sha256 drift")

        tag = "ok " if not bad else "FAIL"
        print(f"  [{tag}] {scheme:14s} train={counts['train']:5d} val={counts['val']:5d} "
              f"test={counts['test']:5d}  sha256={got_hash[:12]}…"
              + (f"   <-- {'; '.join(bad)}" if bad else ""))
        if bad:
            if "sha256 drift" in bad:
                print(f"           expected {info['sha256']}\n           got      {got_hash}")
            ok = False

    print()
    if ok:
        print("PASS — every committed manifest matches its pinned counts + sha256.")
        return 0
    print("FAIL — a manifest drifted. If intentional, re-run make_splits.py to re-pin, then commit.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
