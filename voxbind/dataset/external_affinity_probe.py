#!/usr/bin/env python3
"""Run the existing frozen affinity probe on an external labeled test set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN_FEATURES = (
    HERE
    / "data"
    / "pdbbind"
    / "features"
    / "atomblob_density_gradmag_e99_v5_atomblob7_v2p1_cdg_cvit.pt"
)
DEFAULT_BDB_FEATURES = (
    HERE
    / "data"
    / "bdb2020plus"
    / "features"
    / "atomblob_density_gradmag_e99_v5_bdb2020plus_atomblob7_v2p1.pt"
)
DEFAULT_BDB_LABELS = HERE / "data" / "bdb2020plus" / "index.csv"


def load_probe_module():
    path = HERE / "01c_pdbbind_probe.py"
    spec = importlib.util.spec_from_file_location("pdbbind_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import probe module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_features(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    features = bundle.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError(f"missing feature dictionary in {path}")
    normalized = {str(pid).lower(): tensor.cpu() for pid, tensor in features.items()}
    return normalized, bundle


FEATURE_CONTRACT_KEYS = (
    "condition",
    "epoch",
    "voxel_version",
    "n_in_channels",
    "feature_dim",
    "input_mode",
    "with_gradmag",
    "ligand_radius",
    "atom_source",
    "exp_dir",
)


def feature_contract(bundle: dict) -> dict:
    return {key: bundle.get(key) for key in FEATURE_CONTRACT_KEYS}


def id_hash(pids: list[str]) -> str:
    payload = "\n".join(sorted(pids)).encode()
    return hashlib.sha256(payload).hexdigest()


def load_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    rename = {}
    if "pdb_id" not in labels and "Unnamed: 0" in labels:
        rename["Unnamed: 0"] = "pdb_id"
    if "pK" not in labels and "value" in labels:
        rename["value"] = "pK"
    labels = labels.rename(columns=rename)
    required = {"pdb_id", "pK"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing columns: {sorted(missing)}")
    labels["pdb_id"] = labels["pdb_id"].astype(str).str.lower()
    return labels.dropna(subset=["pK"]).drop_duplicates("pdb_id")


def external_test(
    labels_csv: Path,
    features: dict[str, torch.Tensor],
    test_ids: list[str] | None = None,
) -> tuple[dict, dict]:
    labels = load_labels(labels_csv)
    if test_ids is None:
        n_official = len(labels)
    else:
        wanted = list(dict.fromkeys(str(pid).lower() for pid in test_ids))
        indexed = labels.set_index("pdb_id")
        present = [pid for pid in wanted if pid in indexed.index]
        labels = indexed.loc[present].reset_index()
        n_official = len(wanted)

    labels = labels[labels["pdb_id"].isin(features)]
    if labels.empty:
        raise ValueError("no external labels overlap the external feature bundle")

    pids = labels["pdb_id"].tolist()
    first_dim = int(next(iter(features.values())).numel())
    X = np.stack([features[pid].numpy() for pid in pids]).astype(np.float32)
    if X.shape[1] != first_dim:
        raise ValueError("external feature vectors have inconsistent dimensions")
    test = {
        "X": X,
        "y": labels["pK"].astype(np.float32).to_numpy(),
        "pid": pids,
        "content_hash": id_hash(pids),
        "n_frozen": n_official,
    }
    coverage = {
        "n_official": n_official,
        "n_evaluated": len(pids),
        "test_id_sha256": test["content_hash"],
    }
    return test, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--external_features", type=Path, default=DEFAULT_BDB_FEATURES)
    parser.add_argument("--labels_csv", type=Path, default=DEFAULT_BDB_LABELS)
    parser.add_argument("--test_ids_json", type=Path, default=None,
                        help="Optional JSON containing the official benchmark PDB IDs.")
    parser.add_argument("--test_ids_key", default=None,
                        help="Dictionary key to read from --test_ids_json.")
    parser.add_argument("--benchmark", default="BDB2020+")
    parser.add_argument("--train_split", default="lp_edrscc_v2")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--scatter_csv", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    test_ids = None
    if args.test_ids_json is not None:
        if not args.test_ids_key:
            raise ValueError("--test_ids_key is required with --test_ids_json")
        test_id_payload = json.loads(args.test_ids_json.read_text())
        test_ids = test_id_payload[args.test_ids_key]


    probe = load_probe_module()
    train_features, train_bundle = load_features(args.train_features)
    external_features, external_bundle = load_features(args.external_features)
    train_dim = int(next(iter(train_features.values())).numel())
    external_dim = int(next(iter(external_features.values())).numel())
    if train_dim != external_dim:
        raise ValueError(
            f"feature dimension mismatch: train={train_dim}, external={external_dim}"
        )
    train_contract = feature_contract(train_bundle)
    external_contract = feature_contract(external_bundle)
    if train_contract != external_contract:
        raise ValueError(
            f"feature contract mismatch: train={train_contract}, external={external_contract}"
        )

    split_map, split_scheme = probe.load_frozen_split_map(args.train_split)
    lp_df = probe.load_lp_index(probe.LP_CSV)
    data = probe.build_dataset(
        train_features,
        lp_df,
        drop_covalent=True,
        cl1_only=False,
        split_map=split_map,
    )
    data["test"], coverage = external_test(
        args.labels_csv,
        external_features,
        test_ids=test_ids,
    )

    print(f"=== External frozen affinity probe: {args.benchmark} ===")
    print(f"  train features : {args.train_features}")
    print(f"  test features  : {args.external_features}")
    print(f"  train split    : {split_scheme}")
    print(
        f"  sizes          : train={len(data['train']['pid'])} "
        f"val={len(data['val']['pid'])} test={len(data['test']['pid'])}"
    )
    print(
        f"  coverage       : {coverage['n_evaluated']}/{coverage['n_official']} "
        f"({coverage['n_evaluated'] / coverage['n_official']:.1%})"
    )

    rows = []
    scatter = None
    for seed in range(args.seeds):
        metrics = probe.train_one(
            data,
            seed=seed,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
        )
        pred = metrics.pop("_pred_te")
        y_true = metrics.pop("_yte")
        row = {
            "benchmark": args.benchmark,
            "seed": seed,
            "train_split": split_scheme,
            "train_feature_bundle": str(args.train_features.resolve()),
            "test_feature_bundle": str(args.external_features.resolve()),
            "n_official": coverage["n_official"],
            "n_test_eff": coverage["n_evaluated"],
            "test_id_sha256": coverage["test_id_sha256"],
            **metrics,
        }
        rows.append(row)
        if seed == 0:
            scatter = pd.DataFrame(
                {
                    "benchmark": args.benchmark,
                    "seed": seed,
                    "pid": data["test"]["pid"],
                    "y_true": y_true,
                    "y_pred": pred,
                }
            )
        print(
            f"  seed={seed} test_rho={metrics['test_spearman']:.4f} "
            f"test_r={metrics['test_pearson']:.4f} "
            f"rmse={metrics['test_rmse']:.4f} mae={metrics['test_mae']:.4f}"
        )

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    if args.scatter_csv is not None and scatter is not None:
        args.scatter_csv.parent.mkdir(parents=True, exist_ok=True)
        scatter.to_csv(args.scatter_csv, index=False)

    metric_cols = ["test_spearman", "test_pearson", "test_rmse", "test_mae"]
    summary = {
        metric: {
            "mean": float(output[metric].mean()),
            "std": float(output[metric].std(ddof=1)),
        }
        for metric in metric_cols
    }
    provenance = {
        "benchmark": args.benchmark,
        "train_split": split_scheme,
        "coverage": coverage,
        "train_cache_signature": train_bundle.get("cache_signature"),
        "external_cache_signature": external_bundle.get("cache_signature"),
        "summary": summary,
    }
    provenance_path = args.output_csv.with_suffix(".json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"  results        : {args.output_csv}")
    print(f"  provenance     : {provenance_path}")


if __name__ == "__main__":
    main()
