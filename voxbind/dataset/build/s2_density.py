"""s2_density.py — Stage 2: download experimental density for ligand-MATCHED entries.

Per --dataset (see registry.py / README.md). Fetches the 2Fo-Fc CCP4 / PDBe-EDS maps only
for entries that pass the dataset's match gate (CrossDocked native_filter · PLINDER RSCC≥0.8 ·
PDBbind/MISATO inherent holo), writing the density-availability list. Resumable downloads.

    python dataset/build/s2_density.py --dataset pdbbind          # delegates to 01a density
    python dataset/build/s2_density.py --dataset crossdocked --dry_run
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
    registry.run_stage("s2", args.dataset, passthrough, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
