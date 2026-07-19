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
import re
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

# CleanSplit (GEMS, Nat. Mach. Intell. 2025) released lists — the leakage- and
# redundancy-filtered PDBbind partition. Vendored under splits/cleansplit_ref/.
CLEANSPLIT_DIR = SPLITS_DIR / "cleansplit_ref"
CLEANSPLIT_JSON = CLEANSPLIT_DIR / "PDBbind_data_split_cleansplit.json"
CLEANSPLIT_FOLD = CLEANSPLIT_DIR / "PDBbind_cleansplit_train_val_split_f0.json"

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
    # affinity measurement type — the 'kd/ki' column actually holds Kd / Ki / IC50
    lp["mtype"] = (lp["kd/ki"].astype(str)
                   .str.extract(r"^\s*(Kd|Ki|IC50)", expand=False, flags=re.IGNORECASE)
                   .str.lower())
    # LP-PDBBind per-complex cleaning-level flags (nested: CL3 ⊆ CL2 ⊆ CL1),
    # stored as the strings "True"/"False" in the CSV → coerce to bool.
    for c in ("CL1", "CL2", "CL3"):
        lp[c] = lp[c].astype(str).str.strip().str.lower().isin(["true", "1"])
    q = q.merge(lp[["pid", "year", "mtype", "CL1", "CL2", "CL3"]], on="pid", how="left")
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


def build_lp_edrscc_v2(q: pd.DataFrame) -> tuple[list, dict]:
    """lp_edrscc ∩ affinity measured as Kd or Ki (drop assay-dependent IC50).

    IC50 varies with substrate/enzyme concentration and mechanism, so pIC50 is not
    directly comparable to the thermodynamic pKd/pKi. Restricting to Kd/Ki gives a
    cleaner regression target. Identical ED + (lig & poc RSCC ≥ 0.8) + non-covalent
    quality bar and the same sequence-clustered new_split partition as v1 — just the
    Kd/Ki subset (≈ 3850 / 817 / 1320).
    """
    sub = q[q["new_split"].astype(str).str.lower().isin(SPLIT_NAMES)
            & q["mtype"].isin(["kd", "ki"])]
    pairs = [(p, s.lower()) for p, s in zip(sub["pid"], sub["new_split"].astype(str))]
    prov = {
        "description": "lp_edrscc_v1 ∩ (Kd or Ki) — IC50 dropped; ED-available ∩ (lig & poc RSCC ≥ 0.8), non-covalent",
        "partition": "LP_PDBBind 'new_split' column (sequence-identity clustered)",
        "generalization": "novel-target (sequence-dedup); Kd/Ki-only affinity labels (no assay-dependent IC50)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name,
                   "rscc_threshold": RSCC_THRESHOLD, "measurement_types": ["Kd", "Ki"]},
    }
    return pairs, prov


def build_lp_edrscc_v2_cl(q: pd.DataFrame, n_levels: int) -> tuple[list, dict]:
    """lp_edrscc_v2 further restricted to LP-PDBBind cleaning level(s) CL1..CLn.

    LP-PDBBind ships three nested, per-complex "cleaning levels" (CL3 ⊆ CL2 ⊆ CL1)
    that flag structural / annotation quality with progressively stricter criteria.
    These variants apply the filter to ALL splits (train/val/test), so each is a
    self-contained, progressively-cleaner benchmark that is an EXACT SUBSET of the
    frozen lp_edrscc_v2 partition (identical split assignments — a pid keeps its v2
    bucket). ``n_levels`` ∈ {1,2,3} requires CL1 (and CL2, and CL3) to be True.

    Diverse-split trial: measures how binding-affinity generalization moves as the
    benchmark is cleaned to LP-PDBBind's stricter quality tiers.
    """
    if n_levels not in (1, 2, 3):
        raise ValueError(f"n_levels must be 1, 2 or 3 (got {n_levels})")
    cl_cols = ["CL1", "CL2", "CL3"][:n_levels]
    mask = (q["new_split"].astype(str).str.lower().isin(SPLIT_NAMES)
            & q["mtype"].isin(["kd", "ki"]))
    for c in cl_cols:
        mask = mask & q[c].fillna(False).astype(bool)
    sub = q[mask]
    pairs = [(p, s.lower()) for p, s in zip(sub["pid"], sub["new_split"].astype(str))]

    # Invariant: a CL variant MUST be an exact subset of the frozen v2 manifest
    # (same pids, same split). Guards against any drift in the quality frame vs the
    # committed v2 CSV — if this trips, v2 itself changed and must be re-frozen first.
    v2_csv = SPLITS_DIR / "lp_edrscc_v2.csv"
    if v2_csv.exists():
        v2 = pd.read_csv(v2_csv)
        v2map = dict(zip(v2["pid"].astype(str).str.lower(), v2["split"].astype(str).str.lower()))
        for p, s in pairs:
            if v2map.get(p) != s:
                raise AssertionError(
                    f"CL variant pid {p!r} (split {s!r}) is not in frozen lp_edrscc_v2 "
                    f"with the same split (v2 has {v2map.get(p)!r}) — re-freeze v2 first."
                )

    tag = "+".join(cl_cols)
    prov = {
        "description": f"lp_edrscc_v2 ∩ LP-PDBBind cleaning level(s) {tag} (applied to all splits); "
                       f"ED-available ∩ (lig & poc RSCC ≥ 0.8) ∩ (Kd or Ki), non-covalent",
        "partition": "LP_PDBBind 'new_split' column (sequence-identity clustered)",
        "generalization": f"novel-target (sequence-dedup); Kd/Ki-only; LP cleaning {tag} — a "
                          f"progressively-cleaner subset of lp_edrscc_v2 (all splits filtered)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name,
                   "rscc_threshold": RSCC_THRESHOLD, "measurement_types": ["Kd", "Ki"],
                   "cleaning_levels": cl_cols},
    }
    return pairs, prov


def build_clean_ed(q: pd.DataFrame, indep: bool = False) -> tuple[list, dict]:
    """CleanSplit partition restricted to the lp_edrscc_v2 quality universe.

    Combines two ideas: (1) GEMS' PDBbind CleanSplit (Nat. Mach. Intell. 2025) — a
    leakage-AND-redundancy-filtered PDBbind training set with CASF-2016 as the held-out
    test — and (2) our electron-density + RSCC + Kd/Ki quality bar (the same universe as
    lp_edrscc_v2). This yields a "clean AND density-available" benchmark on which the
    density probe and the external baselines can be compared under CleanSplit's stricter
    de-leaking regime.

    Assignment comes from CleanSplit's RELEASED lists (splits/cleansplit_ref/):
      test  = CASF-2016  (indep=True → the leakage-independent CASF-2016 subset)
      val   = CleanSplit fold-0 validation
      train = CleanSplit filtered train  (minus anything assigned to val/test)
    each intersected with the eligible universe = ED-available ∧ lig&poc RSCC ≥ 0.8 ∧
    non-covalent ∧ (Kd or Ki). To re-derive CleanSplit from scratch (custom thresholds),
    see cleansplit_ref/remove_train_test_sims.py + remove_train_redundancy.py.
    """
    cs = json.loads(CLEANSPLIT_JSON.read_text())
    fold = json.loads(CLEANSPLIT_FOLD.read_text())
    lc = lambda xs: {str(x).strip().lower() for x in xs}

    eligible = set(q.loc[q["mtype"].isin(["kd", "ki"]), "pid"])   # ED+RSCC+KdKi, non-cov
    test_pool = lc(cs["casf2016_indep"] if indep else cs["casf2016"])
    val_pool = lc(fold["validation"])
    train_pool = lc(cs["train"])

    pairs, seen = [], set()
    for pid in sorted(test_pool & eligible):
        pairs.append((pid, "test")); seen.add(pid)
    for pid in sorted(val_pool & eligible):
        if pid not in seen:
            pairs.append((pid, "val")); seen.add(pid)
    for pid in sorted(train_pool & eligible):
        if pid not in seen:
            pairs.append((pid, "train")); seen.add(pid)

    test_desc = "CASF-2016 leakage-independent subset" if indep else "CASF-2016"
    prov = {
        "description": f"GEMS PDBbind CleanSplit (leakage + intra-train redundancy filtered) ∩ "
                       f"ED-available ∩ (lig & poc RSCC ≥ 0.8) ∩ (Kd or Ki), non-covalent; test = {test_desc}",
        "partition": f"CleanSplit released lists (train / fold-0 val / {test_desc})",
        "generalization": "de-leaked (protein+ligand) AND redundancy-reduced training; CASF-2016 held-out test",
        "inputs": {"cleansplit": CLEANSPLIT_JSON.name, "cleansplit_fold": CLEANSPLIT_FOLD.name,
                   "ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name,
                   "rscc_threshold": RSCC_THRESHOLD, "measurement_types": ["Kd", "Ki"],
                   "casf_independent_subset": indep},
        "reference": "Gao/Durairaj et al., 'Resolving data bias improves generalization in binding "
                     "affinity prediction', Nat. Mach. Intell. 2025; code camlab-ethz/GEMS",
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
                    choices=["all", "lp_edrscc_v1", "lp_edrscc_v2",
                             "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123",
                             "time_v1", "misato_md_v1", "clean_ed_v1", "clean_ed_v1_indep"])
    ap.add_argument("--test_from", type=int, default=2018, help="time_v1: first test year (>=)")
    ap.add_argument("--val_year", type=int, default=2017, help="time_v1: validation year")
    args = ap.parse_args()

    manifest_path = SPLITS_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("note",
                        "Frozen, version-controlled splits. Source of truth is the CSV; "
                        "MANIFEST.json pins counts+sha256. Regenerate via make_splits.py.")
    manifest.setdefault("schemes", {})

    want = (["lp_edrscc_v1", "lp_edrscc_v2",
             "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123",
             "time_v1", "misato_md_v1", "clean_ed_v1", "clean_ed_v1_indep"]
            if args.scheme == "all" else [args.scheme])

    # the quality frame is only needed for the pdbbind-derived schemes
    _pdbbind_schemes = ("lp_edrscc_v1", "lp_edrscc_v2",
                        "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123",
                        "time_v1", "clean_ed_v1", "clean_ed_v1_indep")
    q = _load_quality_frame() if any(s in want for s in _pdbbind_schemes) else None

    for scheme in want:
        if scheme == "lp_edrscc_v1":
            pairs, prov = build_lp_edrscc(q)
            fname = "lp_edrscc_v1.csv"
        elif scheme == "lp_edrscc_v2":
            pairs, prov = build_lp_edrscc_v2(q)
            fname = "lp_edrscc_v2.csv"
        elif scheme == "lp_edrscc_v2_cl1":
            pairs, prov = build_lp_edrscc_v2_cl(q, 1)
            fname = "lp_edrscc_v2_cl1.csv"
        elif scheme == "lp_edrscc_v2_cl12":
            pairs, prov = build_lp_edrscc_v2_cl(q, 2)
            fname = "lp_edrscc_v2_cl12.csv"
        elif scheme == "lp_edrscc_v2_cl123":
            pairs, prov = build_lp_edrscc_v2_cl(q, 3)
            fname = "lp_edrscc_v2_cl123.csv"
        elif scheme == "time_v1":
            pairs, prov = build_time(q, args.test_from, args.val_year)
            fname = "time_v1.csv"
        elif scheme == "misato_md_v1":
            pairs, prov = build_misato_md()
            fname = "misato_md_v1.csv"
        elif scheme == "clean_ed_v1":
            pairs, prov = build_clean_ed(q, indep=False)
            fname = "clean_ed_v1.csv"
        elif scheme == "clean_ed_v1_indep":
            pairs, prov = build_clean_ed(q, indep=True)
            fname = "clean_ed_v1_indep.csv"
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
