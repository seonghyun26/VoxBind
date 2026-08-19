"""Re-score CASF-2016 clean-92 on LP-PDBBind protein-novel subsets.

The base cohort removes exact ``lp_edrscc_v2`` train/validation PDB members from the
214 ED/RSCC-evaluable CASF complexes (clean-92).  We then retain test proteins whose
maximum MMseqs2 identity to any ``lp_edrscc_v2`` train protein is strictly below 60%
or 30%, with at least 80% query and target coverage.

No model is retrained.  Saved per-complex CASF predictions are re-scored and written
as schema-A JSON files consumed by ``notebook/html/build_results.py``.
"""

from __future__ import annotations

import csv
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO = Path(__file__).resolve().parents[2]
CASF = REPO / "base/_casf"
OUT = CASF / "casf_similarity"
MMSEQS = REPO / "base/_tools/mmseqs/bin/mmseqs"
POOL_FASTA = REPO / "voxbind/dataset/data/pdbbind/_lba_cluster/lba30/pool.fasta"
LP_SPLIT = REPO / "voxbind/splits/lp_edrscc_v2_cl123.csv"
CASF_MANIFEST = REPO / "voxbind/splits/casf2016_eval.csv"
STRUCTURES = REPO / "voxbind/dataset/data/pdbbind/structures/pbpp-2020"
PIC50_OFFSET = 6.0

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    pid = None
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                pid = line[1:].split()[0].lower()
                sequences[pid] = ""
            elif pid:
                sequences[pid] += line
    return sequences


def pdb_sequence(pid: str) -> str:
    path = STRUCTURES / pid / f"{pid}_protein.pdb"
    seen = set()
    sequence = []
    with path.open() as handle:
        for line in handle:
            if line[:4] != "ATOM":
                continue
            key = (line[21], line[22:26].strip(), line[26])
            if key in seen:
                continue
            seen.add(key)
            sequence.append(AA3TO1.get(line[17:20].strip(), "X"))
    return "".join(sequence)[:1000]


def load_cohorts() -> tuple[dict[str, float], set[str], set[str], dict[str, str]]:
    split = {row["pid"].lower(): row["split"] for row in csv.DictReader(LP_SPLIT.open())}
    train = {pid for pid, part in split.items() if part == "train"}
    rows = list(csv.DictReader(CASF_MANIFEST.open()))
    labels = {row["pid"].lower(): float(row["pK"]) for row in rows}
    clean = {pid for pid in labels if split.get(pid) not in ("train", "val")}
    if len(clean) != 99:
        raise ValueError(f"expected CL3-clean-99, got {len(clean)}")
    return labels, train, clean, split


def maximum_train_identity(train: set[str], clean92: set[str]) -> dict[str, float]:
    sequences = read_fasta(POOL_FASTA)
    needed = train | clean92
    missing = [pid for pid in sorted(needed) if pid not in sequences]
    for pid in missing:
        sequences[pid] = pdb_sequence(pid)
    empty = [pid for pid in needed if not sequences.get(pid)]
    if empty:
        raise ValueError(f"missing protein sequences: {empty}")

    with tempfile.TemporaryDirectory(prefix="casf_lp_similarity_") as tmp_name:
        tmp = Path(tmp_name)
        for name, pids in (("query", clean92), ("target", train)):
            with (tmp / f"{name}.fasta").open("w") as handle:
                for pid in sorted(pids):
                    handle.write(f">{pid}\n{sequences[pid]}\n")
        alignment = tmp / "query_v_train.m8"
        subprocess.run(
            [
                str(MMSEQS), "easy-search", str(tmp / "query.fasta"),
                str(tmp / "target.fasta"), str(alignment), str(tmp / "work"),
                "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
                "-s", "7.5", "--alignment-mode", "3", "--format-output",
                "query,target,fident,alnlen,qcov,tcov", "--threads", "8", "-v", "1",
            ],
            check=True,
        )
        maximum = {pid: 0.0 for pid in clean92}
        with alignment.open() as handle:
            for line in handle:
                query, target, identity, _length, qcov, tcov = line.rstrip().split("\t")[:6]
                query, target = query.lower(), target.lower()
                if (query in maximum and target in train and
                        float(qcov) >= 0.8 and float(tcov) >= 0.8):
                    maximum[query] = max(maximum[query], float(identity))
    return maximum


def jsonl_seeds(pattern: str) -> list[dict[str, float]]:
    seeds = []
    for path in sorted(glob.glob(pattern)):
        pred = {}
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                pred[str(row["id"]).lower()] = float(row["pred"])
        seeds.append(pred)
    return seeds


def csv_seeds(pattern: str, pred_col: str = "pred") -> list[dict[str, float]]:
    return [
        {row["pid"].lower(): float(row[pred_col]) for row in csv.DictReader(open(path))}
        for path in sorted(glob.glob(pattern))
    ]


def hbgsa_seeds() -> list[dict[str, float]]:
    seeds = []
    pattern = REPO / "base/hbgsa/results/preds_casf2016_hbgsa_3p06m_seed*.json"
    for path in sorted(glob.glob(str(pattern))):
        data = json.load(open(path))
        seeds.append({str(pid).lower(): float(value)
                      for pid, value in zip(data["pdb_id"], data["pred"])})
    return seeds


def aev_seeds() -> list[dict[str, float]]:
    rows = list(csv.DictReader((CASF / "AEV_preds.csv").open()))
    return [
        {row["pid"].lower(): float(row[f"pred_seed{seed}"]) for row in rows}
        for seed in range(3)
    ]


def nesso_predictions() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    pred_dir = REPO / "base/nesso/_edrscc/outputs/predictions"
    corr, calibrated = {}, {}
    for directory in pred_dir.iterdir():
        path = directory / "affinity.json"
        if not path.exists():
            continue
        value = json.load(path.open()).get("affinity_pred_value")
        if value is None or not np.isfinite(value):
            continue
        raw = float(value)
        pid = directory.name.lower()
        corr[pid] = -raw
        calibrated[pid] = -raw + PIC50_OFFSET
    return [corr], [calibrated]


LOADERS = {
    "C": lambda: csv_seeds(str(CASF / "C_100m_mask075_coords_casf2016_preds_seed*.csv")),
    "C+D+G": lambda: csv_seeds(str(CASF / "CDG_100m_mask075_casf2016_preds_seed*.csv")),
    "C+D+G +corr": lambda: csv_seeds(str(CASF / "CDG_100m_mask075_corr5_casf2016_preds_seed*.csv")),
    "CheapNet": lambda: jsonl_seeds(str(REPO / "base/cheapnet/_edrscc/preds_casf_seed*.jsonl")),
    "GET": lambda: jsonl_seeds(str(REPO / "base/get/_casf_get/preds/preds_GET_casf_seed*.jsonl")),
    "EGNN": lambda: jsonl_seeds(str(REPO / "base/get/_casf_get/preds/preds_EGNN_casf_seed*.jsonl")),
    "EGNN + TargetDiff": lambda: jsonl_seeds(str(REPO / "base/get/_casf_get/preds/preds_EGNN_TD_casf_seed*.jsonl")),
    "HBGSA": hbgsa_seeds,
    "AEV-PLIG": aev_seeds,
    "ProFSA": lambda: csv_seeds(str(REPO / "base/profsa/_casf/preds/preds_seed*.csv")),
    "BindNet": lambda: csv_seeds(str(CASF / "BindNet_casf2016_preds_seed*.csv")),
    "HonestAffinity": lambda: csv_seeds(str(CASF / "HonestAffinity_casf2016_preds_seed*.csv")),
    "DSMBind": lambda: csv_seeds(str(CASF / "DSMBind_casf2016_preds_seed*.csv")),
    "IPNet (frozen)": lambda: csv_seeds(str(CASF / "IPNet_frozen_casf2016_preds_seed*.csv")),
    "IPNet (scratch)": lambda: csv_seeds(str(CASF / "IPNet_scratch_casf2016_preds_seed*.csv")),
    "DeepDTA": lambda: csv_seeds(str(CASF / "DeepDTA_casf2016_preds_seed*.csv")),
    "MolTrans": lambda: csv_seeds(str(CASF / "MolTrans_casf2016_preds_seed*.csv")),
}


def score_seed(pred: dict[str, float], rmse_pred: dict[str, float], labels: dict[str, float],
               cohort: set[str]) -> dict[str, float]:
    pids = sorted(cohort & pred.keys() & rmse_pred.keys() & labels.keys())
    y = np.asarray([labels[pid] for pid in pids])
    p = np.asarray([pred[pid] for pid in pids])
    pr = np.asarray([rmse_pred[pid] for pid in pids])
    return {
        "n": len(pids),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "rmse": float(np.sqrt(np.mean((pr - y) ** 2))),
    }


def aggregate(method: str, cohort_name: str, cohort: set[str], labels: dict[str, float],
              seeds: list[dict[str, float]], rmse_seeds: list[dict[str, float]] | None = None) -> dict:
    rmse_seeds = rmse_seeds or seeds
    per_seed = [score_seed(pred, rmse_pred, labels, cohort)
                for pred, rmse_pred in zip(seeds, rmse_seeds)]
    if len({row["n"] for row in per_seed}) != 1:
        raise ValueError(f"{method} {cohort_name}: inconsistent coverage {per_seed}")
    result = {
        "method": method,
        "split": cohort_name,
        "n_test": per_seed[0]["n"],
        "seeds": len(per_seed),
        "per_seed": per_seed,
    }
    for metric in ("pearson", "spearman", "rmse"):
        values = [row[metric] for row in per_seed]
        result[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels, train, clean, _split = load_cohorts()
    maximum = maximum_train_identity(train, clean)
    cohorts = {
        "casf_clean_cl3_novel60": {pid for pid, value in maximum.items() if value < 0.60},
        "casf_clean_cl3_novel30": {pid for pid, value in maximum.items() if value < 0.30},
    }
    if len(cohorts["casf_clean_cl3_novel60"]) != 64 or len(cohorts["casf_clean_cl3_novel30"]) != 32:
        raise ValueError(f"unexpected cohort sizes: {[(k, len(v)) for k, v in cohorts.items()]}")

    with (OUT / "clean92_max_lp_train_identity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", "max_lp_train_identity"])
        writer.writerows((pid, maximum[pid]) for pid in sorted(maximum))
    for name, pids in cohorts.items():
        (OUT / f"{name}.txt").write_text("".join(f"{pid}\n" for pid in sorted(pids)))

    loaded = {}
    for method, loader in LOADERS.items():
        seeds = loader()
        if seeds:
            loaded[method] = (seeds, None)
    nesso_corr, nesso_rmse = nesso_predictions()
    if nesso_corr:
        loaded["Nesso-1"] = (nesso_corr, nesso_rmse)

    summary = {
        "reference": "lp_edrscc_v2 train (N=3850)",
        "base_cohort": "CASF-2016 clean-92 (exact lp train/val PDB members removed)",
        "rule": "MMseqs2 fident; qcov>=0.8 and tcov>=0.8; strict < threshold",
        "cohorts": {name: sorted(pids) for name, pids in cohorts.items()},
        "methods": {},
    }
    for method, (seeds, rmse_seeds) in loaded.items():
        summary["methods"][method] = {}
        for cohort_name, pids in cohorts.items():
            result = aggregate(method, cohort_name, pids, labels, seeds, rmse_seeds)
            summary["methods"][method][cohort_name] = result
            path = OUT / f"{safe(method)}__{cohort_name}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"{method:20s} {cohort_name:24s} n={result['n_test']:2d} "
                  f"rho={result['spearman']['mean']:.3f} r={result['pearson']['mean']:.3f} "
                  f"rmse={result['rmse']['mean']:.3f}")
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
