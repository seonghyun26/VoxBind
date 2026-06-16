"""s1_acquire.py — Stage 1: acquire originals + atom-dict filter + split + (pocket,ligand) tuples.

Per --dataset (see registry.py / README.md). Downloads/locates the raw structures or poses,
applies the VoxBind element→channel map (masking unsupported atoms, keeping the complex),
the size filter, the split assignment, and emits the tuple/index manifest.

    python dataset/build/s1_acquire.py --dataset pdbbind          # delegates to 01a structures+index
    python dataset/build/s1_acquire.py --dataset crossdocked --dry_run
Extra args after --dataset/--dry_run are forwarded to the underlying script.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=registry.DATASETS)
    ap.add_argument("--dry_run", action="store_true", help="print the build steps without running them")
    args, passthrough = ap.parse_known_args()
    registry.run_stage("s1", args.dataset, passthrough, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
