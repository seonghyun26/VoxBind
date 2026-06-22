"""consolidate_probe_results.py — Merge scattered PDBbind probe CSVs into one table.

`01c_pdbbind_probe.py probe` writes one CSV per (epoch, voxel_version, variant)
run, e.g.

    probe_results_e99.csv               probe_results_e99_v3.csv
    probe_results_e99_v4_gradmag.csv    probe_results_e99_v1_separate_gradmag.csv
    probe_pocket_only_e99.csv

The density VERSION and run VARIANT live only in the filename, so the in-CSV
`condition` column is NOT a unique key — e.g. atomblob_merged_density seed 0
scores test_ρ 0.295 / 0.346 / 0.423 / 0.253 across v1 / v2 / v3 / v4. This module
folds every per-run CSV into ONE long-form table, lifting that filename metadata
into real columns:

    source_file, epoch, voxel_version, variant, condition, seed, <metrics...>

Older CSVs lack `val_pearson`; it is filled with NaN so every row aligns.

Two entry points
────────────────
  • rebuild()    — glob every per-run CSV and (re)write the consolidated table
                   from scratch. Idempotent: safe to re-run, always overwrites.
  • append_run() — called by 01c_pdbbind_probe.run_probe right after it writes a
                   per-run CSV, so new probes self-consolidate without a manual
                   rebuild. Idempotent per source_file (re-running the same probe
                   replaces that file's rows rather than duplicating them).

A rebuild() reconstructs the exact same table append_run() maintains, so the two
never drift: append keeps it fresh between runs; rebuild is the clean re-sync.

    cd voxbind
    python dataset/consolidate_probe_results.py             # rebuild from all CSVs
    python dataset/consolidate_probe_results.py --summary   # + mean±std pivot
  → dataset/data/pdbbind/probe_results_consolidated.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind" / "results"  # per-run probe CSVs live here
CONSOLIDATED_NAME = "probe_results_consolidated.csv"

# Provenance lifted from the filename + the per-run metrics emitted by the probe.
PROV_COLS = ["source_file", "epoch", "voxel_version", "variant"]
METRIC_COLS = [
    "condition", "seed", "n_train", "n_val", "n_test",
    "best_val_spearman", "val_pearson", "test_spearman", "test_pearson",
    "test_rmse", "epoch_stopped",
]
CANONICAL_COLS = PROV_COLS + METRIC_COLS
SORT_COLS = ["epoch", "voxel_version", "variant", "condition", "seed"]


# ── Filename → metadata ──────────────────────────────────────────────────────
def parse_run_meta(filename: str) -> dict:
    """Extract (epoch, voxel_version, variant) from a probe-result filename.

    The probe names files `probe_results_e{epoch}[_v{N}][_{variant}].csv` (a
    missing `_v{N}` means v1, matching the probe's no-suffix default) plus the
    one-off `probe_pocket_only_e{epoch}.csv`. Examples:
        probe_results_e99                  → e99, v1, "base"
        probe_results_e99_v3               → e99, v3, "base"
        probe_results_e99_v4_gradmag       → e99, v4, "gradmag"
        probe_results_e99_v1_separate_gradmag → e99, v1, "separate_gradmag"
        probe_pocket_only_e99              → e99, v1, "pocket_only"
    """
    stem = filename[:-4] if filename.endswith(".csv") else filename
    toks = stem.split("_")
    is_meta = lambda t: bool(re.fullmatch(r"e\d+", t) or re.fullmatch(r"v\d+", t))
    epoch = next((int(t[1:]) for t in toks if re.fullmatch(r"e\d+", t)), None)
    version = next((t for t in toks if re.fullmatch(r"v\d+", t)), "v1")
    variant_toks = [t for t in toks if t not in ("probe", "results") and not is_meta(t)]
    return {"epoch": epoch, "voxel_version": version,
            "variant": "_".join(variant_toks) if variant_toks else "base"}


def _tag(df: pd.DataFrame, source_file: str, epoch=None,
         voxel_version=None, variant=None) -> pd.DataFrame:
    """Attach provenance columns; explicit args win over the filename parse."""
    meta = parse_run_meta(source_file)
    df = df.copy()
    df["source_file"] = source_file
    df["epoch"] = meta["epoch"] if epoch is None else epoch
    df["voxel_version"] = meta["voxel_version"] if voxel_version is None else voxel_version
    df["variant"] = meta["variant"] if variant is None else variant
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns (canonical first, any extras after) and sort rows."""
    extras = [c for c in df.columns if c not in CANONICAL_COLS]
    df = df.reindex(columns=CANONICAL_COLS + extras)
    present = [c for c in SORT_COLS if c in df.columns]
    return df.sort_values(present, kind="stable", na_position="last").reset_index(drop=True)


# ── Public API ───────────────────────────────────────────────────────────────
def per_run_csvs(pdbbind_dir: Path, out_name: str = CONSOLIDATED_NAME) -> list[Path]:
    """Every per-run probe CSV (both naming families), minus the consolidated one."""
    files = set(pdbbind_dir.glob("probe_results_*.csv")) | set(pdbbind_dir.glob("probe_pocket_only_*.csv"))
    return sorted(f for f in files if f.name != out_name)


def rebuild(pdbbind_dir: Path = DEFAULT_PDBBIND_DIR,
            out_name: str = CONSOLIDATED_NAME, verbose: bool = True) -> pd.DataFrame:
    """Glob all per-run CSVs → one consolidated table; overwrite the output."""
    frames = []
    for f in per_run_csvs(pdbbind_dir, out_name):
        try:
            df = pd.read_csv(f)
        except Exception as e:  # malformed / empty file — note and skip
            if verbose:
                print(f"  [skip] {f.name}: {e!r}")
            continue
        if df.empty or "condition" not in df.columns:
            if verbose:
                print(f"  [skip] {f.name}: no probe rows")
            continue
        frames.append(_tag(df, f.name))
        if verbose:
            m = parse_run_meta(f.name)
            print(f"  [read] {f.name:42s} → {m['voxel_version']:>3} / {m['variant']:<16} ({len(df)} rows)")
    if not frames:
        raise SystemExit(f"no per-run probe CSVs found under {pdbbind_dir}")
    out = _finalize(pd.concat(frames, ignore_index=True, sort=False))
    out_path = pdbbind_dir / out_name
    out.to_csv(out_path, index=False)
    if verbose:
        print(f"[write] {out_path}  ({len(out):,} rows from {len(frames)} files)")
    return out


def append_run(rows_df: pd.DataFrame, source_file: str,
               pdbbind_dir: Path = DEFAULT_PDBBIND_DIR, *, epoch=None,
               voxel_version=None, variant=None,
               out_name: str = CONSOLIDATED_NAME) -> Path:
    """Merge one probe run's rows into the consolidated table (create if absent).

    Idempotent per source_file: any prior rows from the same file are dropped
    first, so re-running a probe replaces — never duplicates — its rows.
    """
    out_path = pdbbind_dir / out_name
    new = _tag(rows_df, source_file, epoch, voxel_version, variant)
    if out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            prev = prev[prev.get("source_file") != source_file]
            combined = pd.concat([prev, new], ignore_index=True, sort=False)
        except Exception:
            combined = new      # unreadable consolidated → start fresh from this run
    else:
        combined = new
    _finalize(combined).to_csv(out_path, index=False)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame) -> None:
    g = (df.groupby(["voxel_version", "variant", "condition"])
           .agg(n_seeds=("seed", "size"),
                n_test=("n_test", "max"),
                test_spearman_mean=("test_spearman", "mean"),
                test_spearman_std=("test_spearman", "std"),
                test_pearson_mean=("test_pearson", "mean"))
           .round(4))
    print("\n── mean±std test ρ per (voxel_version, variant, condition) ──────────")
    print(g.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate scattered PDBbind probe CSVs into one table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pdbbind_dir", default=str(DEFAULT_PDBBIND_DIR))
    ap.add_argument("--out_name", default=CONSOLIDATED_NAME)
    ap.add_argument("--summary", action="store_true",
                    help="also print a mean±std pivot per (version, variant, condition)")
    args = ap.parse_args()
    df = rebuild(Path(args.pdbbind_dir), args.out_name)
    if args.summary:
        _print_summary(df)


if __name__ == "__main__":
    main()
