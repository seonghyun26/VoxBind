"""Leakage primitives for the VoxBind v4 pocket-density corpus.

The protected downstream structures are frozen before corpus construction:

* validation and test PDB IDs from the canonical ``lp_edrscc_v2`` affinity split;
* both the receptor and ligand-source PDB IDs encoded in CrossDocked2020 test
  tuples.

The second point is easy to get wrong. CrossDocked identifiers begin with a
protein-family directory, not a PDB ID. For example::

    BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb

The experimental PDB IDs are ``2z3h`` and ``1wn6``; slicing the first four
characters returns the invalid token ``BSD_``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PDB_ID_PATTERN = r"[0-9][A-Za-z0-9]{3}"
CROSSDOCKED_ID_RE = re.compile(
    rf"(?:^|/)(?P<receptor>{PDB_ID_PATTERN})_"
    rf"(?P<receptor_chain>[^_/]+)_rec_"
    rf"(?P<ligand_source>{PDB_ID_PATTERN})_",
    re.IGNORECASE,
)
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def sha256(path: Path) -> str:
    """Stream a file SHA256 without loading large serialized test sets."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_pdb_id(value: object) -> str:
    """Return a validated lowercase four-character PDB ID."""
    pdb_id = str(value).strip().lower()
    if re.fullmatch(PDB_ID_PATTERN, pdb_id, flags=re.IGNORECASE) is None:
        raise ValueError(f"invalid PDB ID: {value!r}")
    return pdb_id


def parse_crossdocked_id(identifier: object) -> tuple[str, str]:
    """Extract ``(receptor_pdb, ligand_source_pdb)`` from a CrossDocked ID."""
    match = CROSSDOCKED_ID_RE.search(str(identifier))
    if match is None:
        raise ValueError(f"cannot parse CrossDocked PDB IDs from {identifier!r}")
    return (
        normalize_pdb_id(match.group("receptor")),
        normalize_pdb_id(match.group("ligand_source")),
    )


def crossdocked_test_pdbs(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> tuple[set[str], set[str]]:
    """Return receptor and ligand-source PDB sets from serialized test tuples.

    Pocket and ligand identifiers must encode the same experimental pair. A
    mismatch is corruption, not a row to skip.
    """
    receptors: set[str] = set()
    ligand_sources: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"CrossDocked row {index}: expected (pocket, ligand)")
        pocket, ligand = pair
        pocket_ids = parse_crossdocked_id(pocket["id"])
        ligand_ids = parse_crossdocked_id(ligand["id"])
        if pocket_ids != ligand_ids:
            raise ValueError(
                f"CrossDocked row {index}: pocket IDs {pocket_ids} "
                f"!= ligand IDs {ligand_ids}"
            )
        receptors.add(pocket_ids[0])
        ligand_sources.add(pocket_ids[1])
    return receptors, ligand_sources


def read_affinity_split(path: Path) -> dict[str, set[str]]:
    """Read the frozen canonical affinity ``pid,split`` manifest."""
    split_ids = {name: set() for name in SPLIT_PRIORITY}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"pid", "split"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected pid,split columns")
        for row_number, row in enumerate(reader, start=2):
            split = str(row["split"]).strip().lower()
            if split not in split_ids:
                raise ValueError(f"{path}:{row_number}: invalid split {split!r}")
            split_ids[split].add(normalize_pdb_id(row["pid"]))
    overlap = (
        (split_ids["train"] & split_ids["val"])
        | (split_ids["train"] & split_ids["test"])
        | (split_ids["val"] & split_ids["test"])
    )
    if overlap:
        raise ValueError(f"{path}: PDB IDs occur in multiple splits: {sorted(overlap)[:8]}")
    return split_ids


def protected_rows(
    affinity_split: Mapping[str, Iterable[str]],
    crossdocked_receptors: Iterable[str],
    crossdocked_ligand_sources: Iterable[str],
) -> list[dict[str, str]]:
    """Merge external holdouts with conservative ``test > val`` precedence."""
    reasons: dict[str, set[str]] = defaultdict(set)
    requested: dict[str, set[str]] = defaultdict(set)

    for pdb_id in affinity_split["val"]:
        pid = normalize_pdb_id(pdb_id)
        requested[pid].add("val")
        reasons[pid].add("binding_affinity_val")
    for pdb_id in affinity_split["test"]:
        pid = normalize_pdb_id(pdb_id)
        requested[pid].add("test")
        reasons[pid].add("binding_affinity_test")
    for pdb_id in crossdocked_receptors:
        pid = normalize_pdb_id(pdb_id)
        requested[pid].add("test")
        reasons[pid].add("crossdocked2020_test_receptor")
    for pdb_id in crossdocked_ligand_sources:
        pid = normalize_pdb_id(pdb_id)
        requested[pid].add("test")
        reasons[pid].add("crossdocked2020_test_ligand_source")

    rows = []
    for pdb_id in sorted(requested):
        split = max(requested[pdb_id], key=SPLIT_PRIORITY.__getitem__)
        rows.append(
            {
                "pdb_id": pdb_id,
                "required_split": split,
                "reasons": ";".join(sorted(reasons[pdb_id])),
            }
        )
    return rows


def resolve_component_split(
    requested_splits: Iterable[str],
    *,
    default: str = "train",
) -> str:
    """Resolve a merged leakage component with ``test > val > train`` precedence."""
    splits = {str(value).strip().lower() for value in requested_splits if value}
    splits.add(default)
    unknown = splits - set(SPLIT_PRIORITY)
    if unknown:
        raise ValueError(f"unknown component split labels: {sorted(unknown)}")
    return max(splits, key=SPLIT_PRIORITY.__getitem__)


def write_frozen_holdout(
    *,
    affinity_csv: Path,
    crossdocked_pt: Path,
    output_csv: Path,
    output_json: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Build and provenance-stamp the v4 protected-PDB manifest."""
    import torch

    def provenance_path(path: Path) -> str:
        resolved = path.resolve()
        if project_root is not None:
            try:
                return str(resolved.relative_to(project_root.resolve()))
            except ValueError:
                pass
        return str(resolved)

    affinity = read_affinity_split(affinity_csv)
    pairs = torch.load(crossdocked_pt, map_location="cpu", weights_only=False)
    receptors, ligand_sources = crossdocked_test_pdbs(pairs)
    rows = protected_rows(affinity, receptors, ligand_sources)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pdb_id", "required_split", "reasons"]
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["required_split"]] += 1
    metadata: dict[str, object] = {
        "kind": "voxbind_v4_protected_pdb_ids",
        "policy": "PDB protection followed by merged pocket-cluster co-assignment; "
        "test > val > train",
        "sources": {
            "binding_affinity": {
                "path": provenance_path(affinity_csv),
                "sha256": sha256(affinity_csv),
                "counts": {key: len(value) for key, value in affinity.items()},
            },
            "crossdocked2020_test": {
                "path": provenance_path(crossdocked_pt),
                "sha256": sha256(crossdocked_pt),
                "n_pairs": len(pairs),
                "n_receptor_pdbs": len(receptors),
                "n_ligand_source_pdbs": len(ligand_sources),
                "n_unique_pdbs": len(receptors | ligand_sources),
            },
        },
        "counts": {
            "total": len(rows),
            "val": counts["val"],
            "test": counts["test"],
        },
        "csv_sha256": sha256(output_csv),
    }
    output_json.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--affinity-split",
        type=Path,
        default=root / "splits" / "lp_edrscc_v2.csv",
    )
    parser.add_argument(
        "--crossdocked-test",
        type=Path,
        default=root / "dataset" / "data" / "pretrain" / "data_test.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "splits" / "v4",
    )
    args = parser.parse_args()
    metadata = write_frozen_holdout(
        affinity_csv=args.affinity_split,
        crossdocked_pt=args.crossdocked_test,
        output_csv=args.output_dir / "protected_pdb_ids.csv",
        output_json=args.output_dir / "protected_pdb_ids.json",
        project_root=root,
    )
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
