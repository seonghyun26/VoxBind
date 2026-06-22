"""make_splits.py — (re)generate the frozen PDBbind split manifests.

This is the ONLY place a split is *computed*. It reads primary inputs (which live
under the git-ignored ``dataset/data/`` tree) and writes committed ``pid,split``
manifests + ``MANIFEST.json`` (counts, sha256, provenance). Daily train/probe code
never recomputes — it reads the frozen manifest via ``voxbind.splits.load_split``.

Run it intentionally, on a machine with COMPLETE inputs (full RSCC table, etc.),
then commit the regenerated manifests:

    cd /home/shpark/prj-denovo/VoxBind
    python voxbind/splits/make_splits.py                 # all schemes
    python voxbind/splits/make_splits.py --scheme time_v1 --test_from 2018
    git add -f voxbind/splits/*.csv voxbind/splits/MANIFEST.json

Schemes
-------
  lp_edrscc_v1   LP-PDBBind ``new_split`` ∩ ED-available ∩ (lig & poc RSCC ≥ 0.8),
                 non-covalent. The canonical affinity split (≈ 5817/1498/2813).
                 Sequence-identity clustered → measures novel-target generalization.
  time_v1        Temporal holdout over the SAME quality bar (non-cov, lig&poc
                 RSCC ≥ 0.8), partitioned by deposition year: train ≤ 2016,
                 val 2017, test ≥ 2018. Measures temporal generalization.
                 NOTE: does NOT control sequence redundancy (a recent test target
                 may be near-identical to a training one) — interpret accordingly.
  misato_md_v1   Mirror of MISATO's official 8:1:1 MD split (train/val/test_MD.txt),
                 lowercased. Used for MISATO QM/MD targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from voxbind.splits import SPLITS_DIR, SPLIT_NAMES, manifest_hash  # noqa: E402

PDBBIND = REPO_ROOT / "voxbind" / "dataset" / "data" / "pdbbind"
LP_CSV = PDBBIND / "raw" / "LP_PDBBind.csv"
ED_RSCC = PDBBIND / "ed_rscc_split.csv"
MISATO_SPLIT_DIR = REPO_ROOT / "voxbind" / "dataset" / "data" / "misato" / "misato_splits"

RSCC_THRESHOLD = 0.8


# ── primary-input loaders ────────────────────────────────────────────────────
def _load_quality_frame() -> pd.DataFrame:
    """ed_rscc_split rows passing the deterministic quality bar, joined with year.

    Quality bar = non-covalent ∧ lig_rscc ≥ τ ∧ poc_rscc ≥ τ. This is the shared
    universe for both lp_edrscc and time schemes; they differ only in PARTITION.
    """
    if not ED_RSCC.exists():
        raise FileNotFoundError(f"missing primary input: {ED_RSCC}")
    ed = pd.read_csv(ED_RSCC)
    ed["pid"] = ed["pid"].astype(str).str.lower()
    cov = ed["covalent"].astype(str).str.lower().isin(["true", "1"])
    q = ed[(~cov) & (ed["lig_rscc"] >= RSCC_THRESHOLD) & (ed["poc_rscc"] >= RSCC_THRESHOLD)].copy()

    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    lp["year"] = pd.to_datetime(lp["date"], errors="coerce").dt.year
    q = q.merge(lp[["pid", "year"]], on="pid", how="left")
    return q


# ── scheme builders → list[(pid, split)] ─────────────────────────────────────
def build_lp_edrscc(q: pd.DataFrame) -> tuple[list, dict]:
    sub = q[q["new_split"].astype(str).str.lower().isin(SPLIT_NAMES)]
    pairs = [(p, s.lower()) for p, s in zip(sub["pid"], sub["new_split"].astype(str))]
    prov = {
        "description": "LP-PDBBind new_split ∩ ED-available ∩ (lig & poc RSCC ≥ 0.8), non-covalent",
        "partition": "LP_PDBBind 'new_split' column (sequence-identity clustered)",
        "generalization": "novel-target (sequence-dedup train/test)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name, "rscc_threshold": RSCC_THRESHOLD},
    }
    return pairs, prov


def build_time(q: pd.DataFrame, test_from: int, val_year: int) -> tuple[list, dict]:
    yr = q["year"]
    pairs: list[tuple[str, str]] = []
    for pid, y in zip(q["pid"], yr):
        if pd.isna(y):
            continue
        y = int(y)
        if y >= test_from:
            s = "test"
        elif y == val_year:
            s = "val"
        elif y < val_year:
            s = "train"
        else:
            continue  # (year between val_year and test_from, if any) — unused
        pairs.append((pid, s))
    prov = {
        "description": f"temporal holdout: train ≤{val_year - 1}, val {val_year}, test ≥{test_from}; "
                       f"quality bar = non-cov ∧ lig&poc RSCC ≥ {RSCC_THRESHOLD}",
        "partition": f"deposition year (train<{val_year} / val=={val_year} / test>={test_from})",
        "generalization": "temporal (NOT sequence-dedup — recent test target may match a train one)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name, "rscc_threshold": RSCC_THRESHOLD},
    }
    return pairs, prov


def build_misato_md() -> tuple[list, dict]:
    if not MISATO_SPLIT_DIR.exists():
        raise FileNotFoundError(f"missing MISATO split dir: {MISATO_SPLIT_DIR}")
    pairs: list[tuple[str, str]] = []
    for s in SPLIT_NAMES:
        f = MISATO_SPLIT_DIR / f"{s}_MD.txt"
        if not f.exists():
            raise FileNotFoundError(f"missing MISATO split file: {f}")
        for line in f.read_text().splitlines():
            pid = line.strip().lower()
            if pid:
                pairs.append((pid, s))
    prov = {
        "description": "mirror of MISATO official 8:1:1 MD split (train/val/test_MD.txt), lowercased",
        "partition": "MISATO official MD split",
        "generalization": "MISATO MD/QM targets",
        "inputs": {"misato_splits": "data/misato/misato_splits/{train,val,test}_MD.txt"},
    }
    return pairs, prov


def _counts(pairs) -> dict:
    c = {s: 0 for s in SPLIT_NAMES}
    for _, s in pairs:
        c[s] += 1
    return c


def _write_manifest_csv(pairs, path: Path) -> None:
    df = pd.DataFrame(sorted(pairs), columns=["pid", "split"])
    # de-dupe defensively (a pid should appear once); keep first split if collision
    df = df.drop_duplicates(subset="pid", keep="first")
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate frozen PDBbind split manifests.")
    ap.add_argument("--scheme", default="all",
                    choices=["all", "lp_edrscc_v1", "time_v1", "misato_md_v1"])
    ap.add_argument("--test_from", type=int, default=2018, help="time_v1: first test year (>=)")
    ap.add_argument("--val_year", type=int, default=2017, help="time_v1: validation year")
    args = ap.parse_args()

    manifest_path = SPLITS_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("note",
                        "Frozen, version-controlled splits. Source of truth is the CSV; "
                        "MANIFEST.json pins counts+sha256. Regenerate via make_splits.py.")
    manifest.setdefault("schemes", {})

    want = ["lp_edrscc_v1", "time_v1", "misato_md_v1"] if args.scheme == "all" else [args.scheme]

    # the quality frame is only needed for the pdbbind-derived schemes
    q = _load_quality_frame() if any(s in want for s in ("lp_edrscc_v1", "time_v1")) else None

    for scheme in want:
        if scheme == "lp_edrscc_v1":
            pairs, prov = build_lp_edrscc(q)
            fname = "lp_edrscc_v1.csv"
        elif scheme == "time_v1":
            pairs, prov = build_time(q, args.test_from, args.val_year)
            fname = "time_v1.csv"
        elif scheme == "misato_md_v1":
            pairs, prov = build_misato_md()
            fname = "misato_md_v1.csv"
        else:
            raise ValueError(scheme)

        csv_path = SPLITS_DIR / fname
        _write_manifest_csv(pairs, csv_path)
        # re-read what we wrote so counts+hash describe the on-disk artifact exactly
        df = pd.read_csv(csv_path)
        on_disk = list(zip(df["pid"].astype(str), df["split"].astype(str)))
        counts = _counts(on_disk)
        h = manifest_hash(on_disk)
        manifest["schemes"][scheme] = {
            "file": fname,
            "counts": {**counts, "total": sum(counts.values())},
            "sha256": h,
            **prov,
        }
        print(f"[{scheme:14s}] {fname:18s} "
              f"train={counts['train']:5d} val={counts['val']:5d} test={counts['test']:5d}  "
              f"sha256={h[:12]}…")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")
    print("commit with:  git add -f voxbind/splits/*.csv voxbind/splits/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
