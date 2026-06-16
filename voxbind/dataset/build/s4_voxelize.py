"""s4_voxelize.py — Stage 4: voxelize (HYBRID).

Per --dataset (see registry.py / README.md). Atom blobs use the radius modes from the registry
(uniform vs per-element vdW, independently for ligand and pocket).

  - PRECOMPUTE datasets (pdbbind, misato): rasterize atom channels + stack density/gradmag and
    write dense voxel tensors to disk (the probe path; crop is fused in here).
  - ON-THE-FLY datasets (crossdocked, plinder): no precompute — the (pocket,ligand) point-cloud
    tuples ARE the compact form; atoms voxelize on-GPU at train time. s4 is a no-op note.

    python dataset/build/s4_voxelize.py --dataset pdbbind                      # delegates to 01b voxelize+poolnorm
    python dataset/build/s4_voxelize.py --dataset pdbbind --max_complexes 20 --dry_run
Passthrough args are forwarded to EVERY step of the stage, so use only flags common to all steps
of a multi-step stage (e.g. --max_complexes works for both voxelize+poolnorm). The voxelize-only
--ligand_vdw is driven by the registry's ligand_radius (wired during PDBbind parity), not passed here.
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
    ap.add_argument("--audit", action="store_true",
                    help="Classify + report WHY density is missing (reads this stage's availability "
                         "artifact) and write density_audit.json; skips the voxelize run.")
    args, passthrough = ap.parse_known_args()

    if args.audit:
        import density_audit
        density_audit.audit_density(args.dataset)
        return

    registry.run_stage("s4", args.dataset, passthrough, dry_run=args.dry_run)
    # Precompute datasets finalize density availability here → classify missing causes.
    if not args.dry_run:
        import density_audit
        density_audit.audit_density(args.dataset)


if __name__ == "__main__":
    main()
