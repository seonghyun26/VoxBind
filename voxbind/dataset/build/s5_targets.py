"""s5_targets.py — Stage 5: extract downstream supervised labels.

Per --dataset (see registry.py / README.md). The OUTPUT side for the frozen-encoder probes
(separate from the crops/voxels). PDBbind → binding-affinity pK (in index.csv) + B-factors;
MISATO → QM/MD properties (partial charge, interaction energy, RMSF, …). CrossDocked / PLINDER
are self-supervised → no targets.

    python dataset/build/s5_targets.py --dataset misato          # (will delegate to 02b)
    python dataset/build/s5_targets.py --dataset pdbbind --dry_run
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
    registry.run_stage("s5", args.dataset, passthrough, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
