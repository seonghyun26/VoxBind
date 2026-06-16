"""density_audit.py — classify WHY density is missing, per dataset (the s-pipeline's density audit).

Runs automatically at the end of the density-FINALIZING stage (s3 crop/resample for the
on-the-fly datasets; s4 voxelize for the precompute probe sets) and standalone via
`s{3,4}_*.py --dataset X --audit`. Reads the availability artifact that dataset's density
stage produces and writes a missing-cause funnel JSON next to it + prints a summary.

Causes
------
crossdocked (from 00b align, in xray_crops_aligned/{split}_stats.csv):
    no_eds_map · no_deposited_pdb · deposited_parse_failed/deposited_empty · map_load_failed ·
    pocket10_parse_failed · too_few_matched_atoms · weak_density · high_rmsd
    (v6/v7 additionally DROP non-native poses via native_filter — a deliberate selection,
     not a failure; that is NOT a density-availability cause and is reported by the build, not here.)
pdbbind (from 01b voxelize, in pdbbind/voxels/availability.csv): has_density=False (no EDS map) +
    element-filter drops (filter_reason).
plinder (from 03b acquire, in plinder/acquire_cache.json): has_density=False (no PDBe .ccp4).
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402

DATA = registry.DATASET_DIR / "data"


def _funnel(dataset, split, n_total, n_avail, by_cause):
    return {
        "dataset": dataset,
        "split": split,
        "n_total": int(n_total),
        "n_available": int(n_avail),
        "n_missing": int(n_total - n_avail),
        "available_pct": round(100.0 * n_avail / n_total, 1) if n_total else 0.0,
        "missing_by_cause": {k: int(v) for k, v in sorted(by_cause.items(), key=lambda x: -x[1])},
    }


def _audit_crossdocked(splits):
    aligned = DATA / "xray_crops_aligned"
    out = {}
    for split in (splits or ["train", "test"]):
        csvp = aligned / f"{split}_stats.csv"
        if not csvp.exists():
            print(f"  [{split}] no {csvp.name} — run s3 crop/align (or s3 --resample) first")
            continue
        with open(csvp) as fh:
            reasons = [row["reason"] for row in csv.DictReader(fh)]
        c = Counter(reasons)
        ok = c.pop("ok", 0)
        f = _funnel("crossdocked", split, len(reasons), ok, dict(c))
        (aligned / f"{split}_density_audit.json").write_text(json.dumps(f, indent=2))
        out[split] = f
    return out


def _audit_pdbbind(splits):
    avail = DATA / "pdbbind" / "voxels" / "availability.csv"
    if not avail.exists():
        print(f"  no {avail.relative_to(DATA)} — run s4 voxelize first")
        return {}
    rows = list(csv.DictReader(open(avail)))
    truthy = {"true", "1", "yes"}
    n_dens = sum(r.get("has_density", "").lower() in truthy for r in rows)
    by = Counter()
    for r in rows:
        if r.get("has_density", "").lower() not in truthy:
            by["no_eds_map"] += 1
        elif r.get("filtered", "").lower() in truthy:
            by[r.get("filter_reason") or "element_filtered"] += 1
    f = _funnel("pdbbind", "all", len(rows), n_dens, dict(by))
    (DATA / "pdbbind" / "voxels" / "density_audit.json").write_text(json.dumps(f, indent=2))
    return {"all": f}


def _audit_plinder(splits):
    cache = DATA / "plinder" / "acquire_cache.json"
    if not cache.exists():
        print(f"  no {cache.relative_to(DATA)} — run s2 acquire first")
        return {}
    c = json.loads(cache.read_text())
    n_dens = sum(bool(v.get("has_density")) for v in c.values())
    by = Counter()
    for v in c.values():
        if not v.get("has_density"):
            by["no_ccp4_map"] += 1
    f = _funnel("plinder", "all", len(c), n_dens, dict(by))
    (DATA / "plinder" / "density_audit.json").write_text(json.dumps(f, indent=2))
    return {"all": f}


AUDITORS = {
    "crossdocked": _audit_crossdocked,
    "pdbbind": _audit_pdbbind,
    "plinder": _audit_plinder,
    # "misato": TODO — read the QM-built availability + eds_cache once that path is exercised.
}


def audit_density(dataset: str, splits=None, quiet: bool = False) -> dict:
    """Classify missing-density causes for `dataset`; print + write the funnel JSON(s)."""
    fn = AUDITORS.get(dataset)
    if fn is None:
        print(f"  [density audit] {dataset}: not wired yet "
              f"(TODO: read its density-availability artifact).")
        return {}
    res = fn(splits)
    if not quiet:
        for split, f in res.items():
            print(f"  [density audit] {dataset}/{split}: "
                  f"{f['n_available']:,}/{f['n_total']:,} available ({f['available_pct']}%); "
                  f"{f['n_missing']:,} missing —")
            for cause, k in f["missing_by_cause"].items():
                pct = 100.0 * k / f["n_missing"] if f["n_missing"] else 0.0
                print(f"      {cause:24s} {k:>8,}  ({pct:4.1f}% of missing)")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Classify missing-density causes per dataset.")
    ap.add_argument("--dataset", required=True, choices=registry.DATASETS)
    ap.add_argument("--splits", nargs="*", default=None, help="crossdocked: subset of {train,test}")
    audit_density(**{k: v for k, v in vars(ap.parse_args()).items() if v is not None})
