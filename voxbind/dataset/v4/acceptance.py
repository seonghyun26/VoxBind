"""CI acceptance checks for a frozen VoxBind v4 corpus manifest.

The validator implements the hard requirements from the v4 corpus spec. It
fails closed: missing provenance/registration fields are violations, not
warnings. A successful run can also emit the required coverage report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "pdb_id",
    "assembly",
    "chains",
    "uniprot_acc",
    "pocket_id",
    "canonical_pocket_residue_set",
    "cluster_id",
    "split",
    "state",
    "source",
    "structure_origin",
    "mms_id",
    "sample_weight",
    "resolution_A",
    "has_structure_factors",
    "density_path",
    "density_source_pdb_id",
    "grid_frame_id",
    "density_map_R",
    "density_map_t",
    "grid_spacing_A",
    "grid_dim",
    "density_registration_ok",
    "paired_holo_id",
    "paired_apo_id",
    "pocket_rmsd_apo_holo",
    "is_interface_pocket",
}
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_STATES = {"apo", "holo", "holo_alt"}
ALLOWED_SOURCES = {"plinder", "ahoj", "both"}


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _bool_value(value: object) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _density_transform(value_R: object, value_t: object):
    try:
        rotation = np.asarray(value_R, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(value_t, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    values = rotation.tolist()
    for first in range(3):
        for second in range(3):
            product = sum(
                values[first][axis] * values[second][axis]
                for axis in range(3)
            )
            expected = 1.0 if first == second else 0.0
            if abs(product - expected) > 2e-3:
                return None
    determinant = (
        values[0][0]
        * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1]
        * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2]
        * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )
    if abs(determinant - 1.0) > 2e-3:
        return None
    return rotation, translation


def _split_conflicts(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    present = frame[_present(frame[key])]
    counts = present.groupby(key, dropna=False)["split"].nunique()
    bad_keys = set(counts[counts.gt(1)].index)
    return present[present[key].isin(bad_keys)]


def _multi_value_split_conflicts(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    exploded = frame.loc[_present(frame[key]), ["sample_id", "split", key]].copy()
    exploded[key] = exploded[key].astype(str).str.split(";")
    exploded = exploded.explode(key)
    exploded = exploded[exploded[key].astype(str).str.strip().ne("")]
    counts = exploded.groupby(key)["split"].nunique()
    bad_keys = set(counts[counts.gt(1)].index)
    bad_samples = set(exploded.loc[exploded[key].isin(bad_keys), "sample_id"])
    return frame[frame["sample_id"].isin(bad_samples)]


def _pair_table_from_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: set[tuple[str, str, str]] = set()
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        if pd.notna(row.paired_holo_id) and str(row.paired_holo_id).strip():
            rows.add((sample_id, str(row.paired_holo_id), "apo_holo"))
        if pd.notna(row.paired_apo_id) and str(row.paired_apo_id).strip():
            rows.add((sample_id, str(row.paired_apo_id), "apo_holo"))
    return pd.DataFrame(
        sorted(rows), columns=["sample_id_a", "sample_id_b", "relation"]
    )


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    pairs: pd.DataFrame | None = None,
    density_root: Path | None = None,
    check_density_files: bool = False,
) -> dict[str, object]:
    """Run A1-A5 plus MMS and schema guards; return a machine-readable report."""
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    frame = manifest.copy()
    frame["pdb_id"] = frame["pdb_id"].astype(str).str.lower()
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["state"] = frame["state"].astype(str).str.lower()
    frame["source"] = frame["source"].astype(str).str.lower()

    invalid_splits = int((~frame["split"].isin(ALLOWED_SPLITS)).sum())
    invalid_states = int((~frame["state"].isin(ALLOWED_STATES)).sum())
    invalid_sources = int((~frame["source"].isin(ALLOWED_SOURCES)).sum())
    missing_cluster = int((~_present(frame["cluster_id"])).sum())
    duplicate_sample_ids = int(frame["sample_id"].duplicated(keep=False).sum())
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce")
    invalid_sample_weights = int(
        (~weights.map(lambda value: math.isfinite(value) and value > 0)).sum()
    )

    # A1: one complete leakage component in one and only one partition.
    cluster_conflicts = _split_conflicts(frame, "cluster_id")

    # L4: congeneric/matched molecular series cannot straddle partitions.
    mms_conflicts = (
        _multi_value_split_conflicts(frame, "mms_id")
        if "mms_id" in frame.columns
        else frame.iloc[0:0]
    )

    # A2: every explicit apo-holo or holo-holo relationship is co-assigned.
    pair_frame = pairs.copy() if pairs is not None else _pair_table_from_manifest(frame)
    pair_violations = 0
    pair_unknown_samples = 0
    pair_grid_violations = 0
    pair_transform_violations = 0
    if len(pair_frame):
        required_pair_columns = {"sample_id_a", "sample_id_b"}
        missing_pair = required_pair_columns - set(pair_frame.columns)
        if missing_pair:
            raise ValueError(f"pair table missing columns: {sorted(missing_pair)}")
        split_by_sample = frame.set_index("sample_id")["split"].to_dict()
        grid_by_sample = frame.set_index("sample_id")["grid_frame_id"].to_dict()
        transforms_by_sample = {
            row.sample_id: _density_transform(row.density_map_R, row.density_map_t)
            for row in frame.itertuples(index=False)
        }
        for row in pair_frame.itertuples(index=False):
            split_a = split_by_sample.get(row.sample_id_a)
            split_b = split_by_sample.get(row.sample_id_b)
            if split_a is None or split_b is None:
                pair_unknown_samples += 1
            elif split_a != split_b:
                pair_violations += 1
            if (
                row.sample_id_a in grid_by_sample
                and row.sample_id_b in grid_by_sample
                and grid_by_sample[row.sample_id_a] != grid_by_sample[row.sample_id_b]
            ):
                pair_grid_violations += 1
            if hasattr(row, "R_a_to_b") and hasattr(row, "t_a_to_b"):
                transform_a = transforms_by_sample.get(row.sample_id_a)
                transform_b = transforms_by_sample.get(row.sample_id_b)
                edge = _density_transform(row.R_a_to_b, row.t_a_to_b)
                if transform_a is None or transform_b is None or edge is None:
                    pair_transform_violations += 1
                else:
                    map_R_a, map_t_a = transform_a
                    edge_R, edge_t = edge
                    expected_R_b = np.asarray(
                        [
                            [
                                sum(
                                    edge_R[row, axis] * map_R_a[axis, column]
                                    for axis in range(3)
                                )
                                for column in range(3)
                            ]
                            for row in range(3)
                        ]
                    )
                    expected_t_b = np.asarray(
                        [
                            sum(
                                map_t_a[axis] * edge_R[column, axis]
                                for axis in range(3)
                            )
                            + edge_t[column]
                            for column in range(3)
                        ]
                    )
                    map_R_b, map_t_b = transform_b
                    if not (
                        np.allclose(expected_R_b, map_R_b, atol=2e-3)
                        and np.allclose(expected_t_b, map_t_b, atol=5e-2)
                    ):
                        pair_transform_violations += 1

    # A3: cross-source canonicalization leaves one row per structure+pocket.
    duplicate_pockets = int(
        frame.duplicated(["pdb_id", "pocket_id"], keep=False).sum()
    )
    duplicate_residue_sets = int(
        frame.duplicated(
            ["pdb_id", "canonical_pocket_residue_set"], keep=False
        ).sum()
    )

    # A4: every density tensor is experimental, registered, and sourced from
    # the same crystal as its coordinates.
    has_density = _present(frame["density_path"])
    has_sf = frame["has_structure_factors"].map(_bool_value)
    registration_ok = frame["density_registration_ok"].map(_bool_value)
    source_matches = (
        frame["density_source_pdb_id"].astype(str).str.lower().eq(frame["pdb_id"])
    )
    has_grid = _present(frame["grid_frame_id"])
    density_without_sf = int((has_density & ~has_sf).sum())
    density_wrong_source = int((has_density & ~source_matches).sum())
    density_unregistered = int((has_density & (~registration_ok | ~has_grid)).sum())
    invalid_density_transforms = sum(
        _density_transform(row.density_map_R, row.density_map_t) is None
        for row in frame.loc[has_density].itertuples(index=False)
    )
    spacing = pd.to_numeric(frame["grid_spacing_A"], errors="coerce")
    grid_dim = pd.to_numeric(frame["grid_dim"], errors="coerce")
    noncanonical_grid = int(
        (has_density & (~spacing.eq(0.25) | ~grid_dim.eq(64))).sum()
    )
    missing_density_files = 0
    if check_density_files:
        if density_root is None:
            raise ValueError("density_root is required when check_density_files=True")
        missing_density_files = sum(
            not (density_root / str(path)).is_file()
            for path in frame.loc[has_density, "density_path"]
        )

    # A5: no predicted/AF2 structure can carry experimental density.
    predicted = (
        frame["state"].astype(str).str.contains("pred|af2|alphafold", case=False)
        | frame["source"].astype(str).str.contains("pred|af2|alphafold", case=False)
    )
    if "structure_origin" in frame.columns:
        predicted |= frame["structure_origin"].astype(str).str.contains(
            "pred|af2|alphafold", case=False
        )
    predicted_with_density = int((predicted & has_density).sum())

    violations = {
        "invalid_splits": invalid_splits,
        "invalid_states": invalid_states,
        "invalid_sources": invalid_sources,
        "missing_cluster_id": missing_cluster,
        "duplicate_sample_ids": duplicate_sample_ids,
        "invalid_sample_weights": invalid_sample_weights,
        "A1_cluster_split_conflict_rows": int(len(cluster_conflicts)),
        "L4_mms_split_conflict_rows": int(len(mms_conflicts)),
        "A2_pair_split_violations": pair_violations,
        "A2_pair_unknown_samples": pair_unknown_samples,
        "A4_pair_grid_frame_violations": pair_grid_violations,
        "A4_pair_transform_violations": pair_transform_violations,
        "A3_duplicate_pdb_pocket_rows": duplicate_pockets,
        "A3_duplicate_pdb_residue_set_rows": duplicate_residue_sets,
        "A4_density_without_structure_factors": density_without_sf,
        "A4_density_wrong_source_pdb": density_wrong_source,
        "A4_density_unregistered": density_unregistered,
        "A4_invalid_density_map_transform": invalid_density_transforms,
        "A4_noncanonical_voxbind_grid": noncanonical_grid,
        "A4_missing_density_files": missing_density_files,
        "A5_predicted_with_density": predicted_with_density,
    }
    return {
        "ok": not any(violations.values()),
        "n_samples": int(len(frame)),
        "n_pairs": int(len(pair_frame)),
        "violations": violations,
    }


def coverage_report(manifest: pd.DataFrame, validation: dict[str, object]) -> str:
    """Render the A6 coverage report as Markdown."""
    frame = manifest.copy()
    state_counts = frame["state"].value_counts().to_dict()
    split_counts = frame["split"].value_counts().to_dict()
    split_clusters = frame.groupby("split")["cluster_id"].nunique().to_dict()
    density_count = int(_present(frame["density_path"]).sum())
    structure_factor_count = int(
        frame["has_structure_factors"].map(_bool_value).sum()
    )
    apo_pockets = int(frame.loc[frame["state"].eq("apo"), "pocket_id"].nunique())
    total_pockets = int(frame["pocket_id"].nunique())
    interface_count = int(frame["is_interface_pocket"].map(_bool_value).sum())
    mms_ids = {
        token
        for value in frame.loc[_present(frame["mms_id"]), "mms_id"].astype(str)
        for token in value.split(";")
        if token
    }
    training_count = (
        int(frame["in_training_corpus"].map(_bool_value).sum())
        if "in_training_corpus" in frame.columns
        else 0
    )
    loader_val_count = (
        int(frame["in_loader_validation"].map(_bool_value).sum())
        if "in_loader_validation" in frame.columns
        else 0
    )
    rmsd = pd.to_numeric(frame["pocket_rmsd_apo_holo"], errors="coerce").dropna()
    quantiles = (
        {str(q): float(rmsd.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}
        if len(rmsd)
        else {}
    )
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce")
    effective_by_cluster = (
        frame.assign(_sample_weight=weights)
        .groupby("cluster_id")["_sample_weight"]
        .sum()
    )
    effective_by_split = (
        frame.assign(_sample_weight=weights)
        .groupby("split")["_sample_weight"]
        .sum()
        .to_dict()
    )

    lines = [
        "# VoxBind v4 coverage report",
        "",
        "## Frozen methodology",
        "",
        "- Canonical pocket: all protein heavy atoms within 10 Å of the nearest "
        "holo-ligand heavy atom; transfer the same pocket identity to partner states.",
        "- Density grid: each structure's own experimental 2mFo-DFc map, resampled "
        "to 64³ at 0.25 Å/voxel in one common local frame per paired pocket. "
        "Stored rigid transforms map that frame back to each deposited crystal.",
        "- Leakage backbone: PLINDER "
        "`pocket_fident__50__weak__component`; AHoJ rows are admitted only when "
        "their query inherits that backbone. Exact-UniProt, shared-PDB, pair, and "
        "interface-component edges are unioned before splitting; unanchored AHoJ "
        "pockets are excluded from this frozen build.",
        "",
        "## Coverage",
        "",
        f"- Acceptance checks: **{'PASS' if validation['ok'] else 'FAIL'}**",
        f"- Samples: **{len(frame):,}**",
        f"- Canonical pockets: **{total_pockets:,}**",
        f"- Pockets with apo state: **{apo_pockets:,}**",
        f"- Samples passing the structure-factor gate: "
        f"**{structure_factor_count:,}**",
        f"- Density-backed samples: **{density_count:,}**",
        f"- Interface-pocket samples: **{interface_count:,}**",
        f"- PLINDER matched molecular series represented: **{len(mms_ids):,}**",
        f"- Materialized training tuples: **{training_count:,} train + "
        f"{loader_val_count:,} validation**",
        "",
        "## State counts",
        "",
        "```json",
        json.dumps(state_counts, indent=2),
        "```",
        "",
        "## Split sample counts",
        "",
        "```json",
        json.dumps(split_counts, indent=2),
        "```",
        "",
        "## Split cluster counts",
        "",
        "```json",
        json.dumps(split_clusters, indent=2),
        "```",
        "",
        "## Apo-holo pocket RMSD quantiles (Å)",
        "",
        "```json",
        json.dumps(quantiles, indent=2),
        "```",
        "",
        "## Redundancy-reweighted counts",
        "",
        "Effective sample weight by split:",
        "",
        "```json",
        json.dumps(effective_by_split, indent=2),
        "```",
        "",
        f"- Effective total sample weight: **{weights.sum():,.2f}**",
        f"- Median effective weight per cluster: **{effective_by_cluster.median():,.3f}**",
        f"- Maximum effective weight per cluster: **{effective_by_cluster.max():,.3f}**",
        "",
        "## Acceptance violations",
        "",
        "```json",
        json.dumps(validation["violations"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--density-root", type=Path)
    parser.add_argument("--check-density-files", action="store_true")
    parser.add_argument("--coverage-report", type=Path)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    pairs = pd.read_parquet(args.pairs) if args.pairs else None
    result = validate_manifest(
        manifest,
        pairs=pairs,
        density_root=args.density_root,
        check_density_files=args.check_density_files,
    )
    if args.coverage_report:
        args.coverage_report.write_text(coverage_report(manifest, result))
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
