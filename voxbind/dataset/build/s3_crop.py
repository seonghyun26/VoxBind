"""s3_crop.py — Stage 3: align + crop pocket/ligand box + normalize density.

Per --dataset (see registry.py / README.md). Frame alignment is dataset-conditional:
CrossDocked needs residue-matched Kabsch (docking→crystal frame); PDBbind/MISATO/PLINDER are
deposited-frame (transform=None, no Kabsch). Then crop a 64³ × 0.25 Å box at the ligand
centroid, normalize density to the canonical arcsinh+z scale, and gate availability.

For PRECOMPUTE datasets (pdbbind, misato) the crop is fused into the voxelize precompute
(stage 4) by the underlying script, so s3 is a no-op note for them.

    python dataset/build/s3_crop.py --dataset crossdocked        # delegates to 00b (--version vN)
    python dataset/build/s3_crop.py --dataset pdbbind --dry_run
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
    ap.add_argument("--resample", action="store_true",
                    help="On-the-fly density: emit a manifest + normalization recipe and KEEP the maps "
                         "(no frozen crops). Density is cropped from the full map at the augmented pose "
                         "at train time. Requires the dataset's density_mode == 'resample_map'.")
    ap.add_argument("--audit", action="store_true",
                    help="Classify + report WHY density is missing for this dataset (reads the density "
                         "stage's availability artifact) and write {split}_density_audit.json; skips the crop run.")
    args, passthrough = ap.parse_known_args()

    if args.audit:
        import density_audit
        density_audit.audit_density(args.dataset)
        return

    if args.resample:
        spec = registry.SPEC[args.dataset]
        if spec.density_mode != "resample_map":
            raise SystemExit(
                f"--resample not supported for {args.dataset!r} (density_mode={spec.density_mode!r}); "
                f"its s3 script has no resample mode. Supported: "
                f"{[d for d in registry.DATASETS if registry.SPEC[d].density_mode == 'resample_map']}.")
        # forward to the resample-capable s3 script (00b --mode resample)
        passthrough = ["--mode", "resample", *passthrough]
        print("[s3 --resample] on-the-fly density prep: writes manifest + recipe, keeps maps, NO crops.")
        print("                NB the full CCP4 maps must be retained on disk for train-time resampling.")

    registry.run_stage("s3", args.dataset, passthrough, dry_run=args.dry_run)

    # After a real density run, classify why crops are missing (no map / weak density / …).
    if not args.dry_run:
        import density_audit
        density_audit.audit_density(args.dataset)


if __name__ == "__main__":
    main()
