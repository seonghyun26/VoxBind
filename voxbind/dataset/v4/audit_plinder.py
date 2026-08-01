"""Audit the PLINDER v3 seed against v4 component-level leakage rules.

This is a metadata-only preflight. It does not build coordinates or density.
The leakage unit is PLINDER's 50% pocket-identity weak component:
``pocket_fident__50__weak__component``. Test takes precedence over validation,
which takes precedence over training.

Two sources can mark a complete component as held out:

1. PLINDER's own seed test/validation rows;
2. downstream protected PDB experiments frozen by :mod:`dataset.v4.leakage`.

The output reports how many v3 holo systems remain eligible for representation
pretraining after propagating both sources to complete components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

try:
    from .leakage import sha256
except ImportError:  # direct ``python dataset/v4/audit_plinder.py``
    from leakage import sha256

DEFAULT_CLUSTER_COLUMN = "pocket_fident__50__weak__component"


def unique_annotations(path: Path, cluster_column: str) -> pd.DataFrame:
    """Load one stable PDB/component mapping per PLINDER system."""
    frame = pq.read_table(
        path,
        columns=["system_id", "entry_pdb_id", cluster_column],
    ).to_pandas()
    grouped = frame.groupby("system_id", sort=False)
    ambiguous_pdb = grouped["entry_pdb_id"].nunique().gt(1)
    ambiguous_cluster = grouped[cluster_column].nunique().gt(1)
    if ambiguous_pdb.any() or ambiguous_cluster.any():
        raise ValueError(
            "PLINDER annotation duplicates disagree: "
            f"PDB={int(ambiguous_pdb.sum())}, "
            f"component={int(ambiguous_cluster.sum())}"
        )
    frame = frame.drop_duplicates("system_id").copy()
    frame["pdb_id"] = frame["entry_pdb_id"].astype(str).str.lower()
    return frame[["system_id", "pdb_id", cluster_column]]


def component_assignments(
    annotations: pd.DataFrame,
    plinder_split: pd.DataFrame,
    protected: pd.DataFrame,
    cluster_column: str,
) -> tuple[set[str], set[str]]:
    """Return disjoint ``(test_components, val_components)``."""
    seeded = plinder_split.merge(
        annotations[["system_id", cluster_column]],
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    if seeded[cluster_column].isna().any():
        raise ValueError("PLINDER split contains systems missing from annotations")

    protected_map = annotations.merge(
        protected[["pdb_id", "required_split"]],
        on="pdb_id",
        how="inner",
    )
    test_components = set(
        seeded.loc[seeded["split"].eq("test"), cluster_column]
    ) | set(
        protected_map.loc[
            protected_map["required_split"].eq("test"), cluster_column
        ].dropna()
    )
    val_components = (
        set(seeded.loc[seeded["split"].eq("val"), cluster_column])
        | set(
            protected_map.loc[
                protected_map["required_split"].eq("val"), cluster_column
            ].dropna()
        )
    ) - test_components
    if test_components & val_components:
        raise AssertionError("test and validation component sets overlap")
    return test_components, val_components


def build_report(
    *,
    annotation_path: Path,
    plinder_split_path: Path,
    v3_selection_path: Path,
    protected_path: Path,
    cluster_column: str,
) -> dict[str, object]:
    annotations = unique_annotations(annotation_path, cluster_column)
    seed = pq.read_table(
        plinder_split_path, columns=["system_id", "split"]
    ).to_pandas()
    protected = pd.read_csv(protected_path)
    selected = pd.read_csv(v3_selection_path, usecols=["system_id"])

    test_components, val_components = component_assignments(
        annotations, seed, protected, cluster_column
    )
    selected = selected.merge(
        annotations[["system_id", cluster_column]],
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    if selected[cluster_column].isna().any():
        raise ValueError("v3 selection contains systems missing from annotations")

    selected["v4_split"] = "train"
    selected.loc[selected[cluster_column].isin(val_components), "v4_split"] = "val"
    selected.loc[selected[cluster_column].isin(test_components), "v4_split"] = "test"

    split_component_sets = {
        split: set(selected.loc[selected["v4_split"].eq(split), cluster_column])
        for split in ("train", "val", "test")
    }
    intersections = {
        "train_val": len(split_component_sets["train"] & split_component_sets["val"]),
        "train_test": len(split_component_sets["train"] & split_component_sets["test"]),
        "val_test": len(split_component_sets["val"] & split_component_sets["test"]),
    }
    if any(intersections.values()):
        raise AssertionError(f"v4 component leakage: {intersections}")

    return {
        "kind": "voxbind_v4_plinder_component_audit",
        "cluster_column": cluster_column,
        "split_precedence": "test > val > train",
        "inputs": {
            "annotation_table": {
                "path": str(annotation_path),
                "sha256": sha256(annotation_path),
            },
            "plinder_split": {
                "path": str(plinder_split_path),
                "sha256": sha256(plinder_split_path),
            },
            "v3_selection": {
                "path": str(v3_selection_path),
                "sha256": sha256(v3_selection_path),
            },
            "protected_pdb_ids": {
                "path": str(protected_path),
                "sha256": sha256(protected_path),
            },
        },
        "components": {
            "all_annotation": int(annotations[cluster_column].nunique()),
            "annotation_systems_missing_component": int(
                annotations[cluster_column].isna().sum()
            ),
            "test": len(test_components),
            "val": len(val_components),
        },
        "v3_systems": {
            key: int(value)
            for key, value in selected["v4_split"].value_counts().to_dict().items()
        },
        "v3_components": {
            key: len(value) for key, value in split_component_sets.items()
        },
        "component_intersections": intersections,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "dataset" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-table",
        type=Path,
        default=data / "plinder" / "index" / "annotation_table.parquet",
    )
    parser.add_argument(
        "--plinder-split",
        type=Path,
        default=data / "plinder" / "splits" / "split.parquet",
    )
    parser.add_argument(
        "--v3-selection",
        type=Path,
        default=root / "splits" / "plinder" / "v3" / "plinder_selected.csv",
    )
    parser.add_argument(
        "--protected",
        type=Path,
        default=root / "splits" / "v4" / "protected_pdb_ids.csv",
    )
    parser.add_argument("--cluster-column", default=DEFAULT_CLUSTER_COLUMN)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "splits" / "v4" / "plinder_v3_component_audit.json",
    )
    args = parser.parse_args()
    report = build_report(
        annotation_path=args.annotation_table,
        plinder_split_path=args.plinder_split,
        v3_selection_path=args.v3_selection,
        protected_path=args.protected,
        cluster_column=args.cluster_column,
    )
    root_resolved = root.resolve()
    for source in report["inputs"].values():
        source_path = Path(source["path"]).resolve()
        try:
            source["path"] = str(source_path.relative_to(root_resolved))
        except ValueError:
            source["path"] = str(source_path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["v3_systems"], indent=2))


if __name__ == "__main__":
    main()
