"""verify_split.py — DEPRECATED shim → voxbind/splits/verify.py

This used to regenerate the canonical PDBbind split from `availability.csv` and assert
the old 2172/480/839 counts. That approach was the *cause* of cross-server drift: the
split was recomputed from server-local, git-ignored artifacts (which crops/RSCC built
here), so two machines silently scored on different complexes.

Splits are now **frozen, committed manifests** under `voxbind/splits/` (one `pid,split`
CSV per scheme, pinned by sha256 in MANIFEST.json). The canonical affinity split is
`lp_edrscc_v1` (5817/1498/2813). This shim just delegates to the new hash gate so the
old entrypoint keeps working.

    cd /home/shpark/prj-denovo/VoxBind/voxbind
    python dataset/verify_split.py            # → voxbind/splits/verify.py (all schemes)

See voxbind/splits/README.md.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    print("[deprecated] dataset/verify_split.py → voxbind/splits/verify.py "
          "(frozen manifests; canonical = lp_edrscc_v1 5817/1498/2813)\n")
    from voxbind.splits import verify
    return verify.main([])          # run the full gate; stale flags (--no-hash …) are gone


if __name__ == "__main__":
    sys.exit(main())
