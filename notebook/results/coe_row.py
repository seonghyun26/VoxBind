"""CoE row, heavy-atom medians and n corrections for the task-2 LaTeX table.

WHY THIS EXISTS: `drug_design_vina_to_latex` in latex-task2.ipynb builds every row from
Table 1 of `results/reports/results_drug_design.html`. Three things are wrong or missing
there, and all three are recovered from the bundle the collector writes:

  * the CoE arm (\\oursC) has no row at all -- that report predates its docking, which
    finished 2026-09-15 (79/79 pockets, 5,910 of 5,911 molecules docked);
  * the "# atoms / mol" MEDIAN is carried as a `data-med` attribute on the BASELINE rows
    only, so Reference, TargetDiff, VoxBind sigma=0.9, \\ours and \\oursC printed "-";
  * the VoxBind sigma=0.9 row reports n=7,887, while every bundle source says 7,883.

EVERY VALUE IS TAKEN OVER THE MOLECULE SET ITS ROW REPORTS -- pairing a 79-pocket row with a
100-pocket median would be wrong, and both Reference and TargetDiff carry both slices
(TargetDiff: density79 median 22.0 over 7,287 molecules vs all 23.0 over 9,206).

COLUMN ORDER is Table 1's, i.e. what the converter calls `vals` (16 entries):
    0-5    Vina Score / Min / Dock, avg then median   (molecule-pooled)
    6-7    High affinity %, avg then median           (per-pocket shares)
    8-13   QED / SA / Diversity, avg then median
    14     heavy atoms, avg
    15     n molecules
The converter reorders to put `n` first and appends the heavy-atom median column.

DIVERSITY MEDIAN: `collect_task2_eval.py` stores a single `diversity` per arm (the mean over
pockets) and no median, so the median is recomputed from per-molecule SMILES with RDKit
fingerprints -- the fingerprint this project fixed for diversity. The self-test re-derives
CoDE's published row from the same code path, so a wrong fingerprint or a wrong column
mapping fails loudly instead of silently shifting a number.
"""
from __future__ import annotations

import json
import statistics as st
from functools import lru_cache
from pathlib import Path

T2 = Path("results/task2-drugdesign")
CODE_ARM = "VoxBind-Ours"       # CoDE (VoxBind + CDG)
COE_ARM = "VoxBind+CoE"         # CoE  (VoxBind + C)
VANILLA_ARM = "VoxBind-vanilla"
TARGETDIFF_ARM = "TargetDiff"   # staged into the bundle 2026-09-15 (see its SOURCE.txt)
# The name the converter's NAME map turns into \;+ \oursC{}; also what marks the row as ours.
COE_ROW_NAME = "Ours · coords"
# HTML row name -> (bundle arm, which pocket set that row reports)
ROW_SOURCE = {
    "Ours · v1": (CODE_ARM, "all"),            # 7,873 molecules, 79 pockets
    COE_ROW_NAME: (COE_ARM, "all"),            # 5,911 molecules, 79 pockets
    "Reference ligand": ("Reference", "density79"),
    "VoxBind σ=0.9": (VANILLA_ARM, "density79"),
    "TargetDiff": (TARGETDIFF_ARM, "density79"),
}


def _eval_sets(repo: Path, arm: str, evaluation: str) -> dict:
    """`pocket_sets` for one evaluation, or {"all": <flat envelope>} for older layouts."""
    path = repo / T2 / arm / "eval" / evaluation / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run voxbind/scripts/tools/collect_task2_eval.py first")
    data = json.loads(path.read_text())
    return data.get("pocket_sets") or {"all": data}


def _eval_block(repo: Path, arm: str, evaluation: str, prefer=("all", "density79")) -> dict:
    """One evaluation's aggregate block, tolerating the collector's two key layouts.

    sample_quality / posecheck / posebusters key their set "all"; the Vina fallback for a
    native arm keys it "density79" (collect_native, eval_docking_results_full79.json).
    Reading only "all" is how the Vina numbers silently came back None once. `prefer` lets a
    caller demand the 79-pocket slice for an arm that also carries the 100-pocket one.
    """
    sets = _eval_sets(repo, arm, evaluation)
    for key in prefer:
        if key in sets:
            return sets[key]
    return next(iter(sets.values()))


@lru_cache(maxsize=8)
def _pocket_diversity(repo_str: str, arm: str) -> tuple[float, float]:
    """(mean, median) of per-pocket diversity = 1 - mean pairwise Tanimoto, RDKit FP."""
    from rdkit import Chem, DataStructs, RDLogger

    RDLogger.DisableLog("rdApp.*")
    values = []
    samples = Path(repo_str) / T2 / arm / "samples"
    for metrics in sorted(samples.glob("target_*/metrics.json")):
        try:
            rows = json.loads(metrics.read_text()).get("samples", [])
        except (json.JSONDecodeError, OSError):
            continue
        mols = [Chem.MolFromSmiles(r["smiles"]) for r in rows if r.get("smiles")]
        mols = [m for m in mols if m is not None]
        if len(mols) < 2:
            continue
        fps = [Chem.RDKFingerprint(m) for m in mols]
        sims = []
        for i in range(len(fps)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        values.append(1.0 - sum(sims) / len(sims))
    if not values:
        raise ValueError(f"no per-molecule SMILES under {samples}")
    return st.mean(values), st.median(values)


def arm_vals(repo: Path, arm: str) -> list[str]:
    """Table 1's 16 values for `arm`, formatted exactly as the HTML carries them."""
    vina = _eval_block(repo, arm, "vina_docking")
    qual = _eval_block(repo, arm, "sample_quality")
    div_mean, div_median = _pocket_diversity(str(repo), arm)

    # The collector's own diversity is the mean over pockets; if the recomputation disagrees
    # the fingerprint or the molecule set has drifted, and the row would be wrong.
    stored = qual.get("diversity")
    if stored is not None and abs(stored - div_mean) > 0.005:
        raise ValueError(
            f"{arm}: diversity mismatch — collector {stored:.4f} vs recomputed {div_mean:.4f}"
        )

    def pair(key: str) -> list[str]:
        block = vina.get(key) or {}
        return ["%.2f" % block["mean"], "%.2f" % block["median"]]

    return [
        *pair("vina_score"), *pair("vina_min"), *pair("vina_dock"),
        "%.1f" % (100 * vina["high_affinity"]),
        "%.1f" % (100 * vina["high_affinity_median"]),
        "%.2f" % qual["qed_mean"], "%.2f" % qual["qed_median"],
        "%.2f" % qual["sa_mean"], "%.2f" % qual["sa_median"],
        "%.2f" % div_mean, "%.2f" % div_median,
        "%.1f" % qual["n_atoms_mean"],
        "{:,}".format(qual["n_molecules"]),
    ]


def coe_entry(repo: Path) -> dict:
    """The CoE row in the shape `drug_design_vina_to_latex` collects from the HTML."""
    return {"kind": "data", "name": COE_ROW_NAME, "vals": arm_vals(repo, COE_ARM),
            "heavy_med": "-"}


def _quality(repo: Path, row_name: str) -> dict:
    arm, pocket_set = ROW_SOURCE[row_name]
    return _eval_block(repo, arm, "sample_quality", prefer=(pocket_set,))


def heavy_medians(repo: Path) -> dict[str, str]:
    """HTML row name -> "# atoms / mol" MEDIAN, for the rows the HTML leaves empty.

    A source that is missing drops out of the mapping, so the converter falls back to the
    HTML's data-med and then to "-" -- never to a number from a different pocket set.
    """
    out: dict[str, str] = {}
    for row_name in ROW_SOURCE:
        try:
            out[row_name] = "%.1f" % _quality(repo, row_name)["n_atoms_median"]
        except Exception:
            pass
    return out


def n_overrides(repo: Path) -> dict[str, str]:
    """HTML row name -> corrected molecule count, where the HTML disagrees with the bundle.

    Only VoxBind sigma=0.9 needs this: the report says 7,887 while the bundle's 79-pocket
    slice has 7,883 in both sample_quality and vina_docking. The value is read from the
    bundle rather than hardcoded, so it follows the data if the arm is ever re-scored.
    """
    out: dict[str, str] = {}
    try:
        out["VoxBind σ=0.9"] = "{:,}".format(_quality(repo, "VoxBind σ=0.9")["n_molecules"])
    except Exception:
        pass
    return out


def _self_test(repo: Path) -> None:
    """CoDE's published row must come back out of this code path, value for value."""
    expected = ["-6.58", "-7.20", "-7.64", "-7.70", "-8.48", "-8.51", "68.1", "78.0",
                "0.52", "0.53", "0.68", "0.68", "0.71", "0.70", "24.9", "7,873"]
    got = arm_vals(repo, CODE_ARM)
    bad = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    if bad:
        raise AssertionError(f"CoDE row mismatch at columns {bad}")
    print("self-test OK — CoDE row reproduced from the bundle:", " ".join(got))

    # Each row's median must come from the molecule set that row reports.
    for row_name, n_expected in (("TargetDiff", 7287), ("Ours · v1", 7873),
                                 (COE_ROW_NAME, 5911), ("Reference ligand", 79),
                                 ("VoxBind σ=0.9", 7883)):
        n_got = _quality(repo, row_name)["n_molecules"]
        if n_got != n_expected:
            raise AssertionError(f"{row_name}: {n_got:,} molecules, expected {n_expected:,}")
    print("self-test OK — every row's pocket set matches its reported n")


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    _self_test(root)
    print("CoE row values:", " ".join(arm_vals(root, COE_ARM)))
    print("heavy-atom medians:", heavy_medians(root))
    print("n overrides:", n_overrides(root))
