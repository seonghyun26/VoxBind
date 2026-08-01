"""Stream AHoJ v2c apo/holo metadata into a compact Parquet relation table.

The 37.42 GiB download is an outer ``tar.gz`` containing a nested
``data.tar.gz``. This reader never expands either archive. It walks the nested
stream once and retains only:

* the query holo row from ``global_results.csv``;
* apo rows from ``apo_filtered_sorted_results.csv``;
* alternate-holo rows from ``holo_filtered_sorted_results.csv``.

Alignment matrices, logs, AlphaFold classifications, and other intermediate
files remain compressed. Coordinates and density are deliberately not fetched
here; canonical 10 Å VoxBind pockets, cross-source deduplication, leakage
assignment, and the structure-factor gate come first.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .ahoj_archive import verify_archive
except ImportError:  # direct ``python dataset/v4/ingest_ahoj.py``
    from ahoj_archive import verify_archive

RELATION_FILES = {
    "apo_filtered_sorted_results.csv": "apo",
    "holo_filtered_sorted_results.csv": "holo_alt",
    "global_results.csv": "holo",
}
SCHEMA = pa.schema(
    [
        ("relation_id", pa.string()),
        ("ahoj_query_id", pa.string()),
        ("query_pdb_id", pa.string()),
        ("query_chain", pa.string()),
        ("query_ligand_ccd", pa.string()),
        ("query_ligand_residue", pa.string()),
        ("target_pdb_id", pa.string()),
        ("target_pocket_index", pa.string()),
        ("state", pa.string()),
        ("query_poi", pa.string()),
        ("role", pa.string()),
        ("apoholo_assignment", pa.string()),
        ("uniprot_acc", pa.string()),
        ("chains2", pa.string()),
        ("chains3", pa.string()),
        ("chains4", pa.string()),
        ("resolution_A", pa.float32()),
        ("experimental_method", pa.string()),
        ("mapped_sequence_percent", pa.float32()),
        ("mapped_observed_percent", pa.float32()),
        ("unobserved_residues", pa.int32()),
        ("tm_score", pa.float32()),
        ("tm_score_inverse", pa.float32()),
        ("alignment_rmsd_A", pa.float32()),
        ("alignment_length", pa.int32()),
        ("pocket_rmsd_A", pa.float32()),
        ("pocket_length", pa.int32()),
        ("pocket_distance_A", pa.float32()),
        ("alignment_matrix", pa.string()),
        ("radius_of_gyration_distance_A", pa.float32()),
        ("source_member", pa.string()),
    ]
)


def text(value: object) -> str | None:
    stripped = str(value or "").strip()
    return stripped or None


def number(value: object, cast=float):
    value = text(value)
    if value is None or value == "-":
        return None
    try:
        return cast(float(value))
    except (TypeError, ValueError):
        return None


def parse_query_id(query_id: str) -> tuple[str, str, str, str]:
    fields = query_id.split("-", 3)
    if len(fields) != 4:
        raise ValueError(f"invalid AHoJ query directory: {query_id!r}")
    pdb_id, chain, ligand_ccd, ligand_residue = fields
    if len(pdb_id) != 4:
        raise ValueError(f"invalid AHoJ query PDB ID: {query_id!r}")
    return pdb_id.lower(), chain, ligand_ccd, ligand_residue


def relation_row(
    row: dict[str, str],
    *,
    query_id: str,
    state: str,
    source_member: str,
) -> dict[str, object]:
    query_pdb, query_chain, ligand_ccd, ligand_residue = parse_query_id(query_id)
    target_pdb = str(row.get("structure", "")).strip().lower()
    target_pocket = str(row.get("pocket", "")).strip()
    relation_id = f"ahoj:{query_id}:{target_pdb}:{target_pocket}:{state}"
    return {
        "relation_id": relation_id,
        "ahoj_query_id": query_id,
        "query_pdb_id": query_pdb,
        "query_chain": query_chain,
        "query_ligand_ccd": ligand_ccd,
        "query_ligand_residue": ligand_residue,
        "target_pdb_id": target_pdb,
        "target_pocket_index": target_pocket,
        "state": state,
        "query_poi": text(row.get("query_POI")),
        "role": text(row.get("role")),
        "apoholo_assignment": text(row.get("apoholo_assignment")),
        "uniprot_acc": text(row.get("UNPs")),
        "chains2": text(row.get("chains2")),
        "chains3": text(row.get("chains3")),
        "chains4": text(row.get("chains4")),
        "resolution_A": number(row.get("resolution")),
        "experimental_method": text(row.get("exp_method")),
        "mapped_sequence_percent": number(row.get("%mapped_sqr")),
        "mapped_observed_percent": number(row.get("%mapped_obs")),
        "unobserved_residues": number(row.get("unobserved_rsd"), int),
        "tm_score": number(row.get("tm_score")),
        "tm_score_inverse": number(row.get("tm_score_i")),
        "alignment_rmsd_A": number(row.get("rmsd")),
        "alignment_length": number(row.get("aln_len"), int),
        "pocket_rmsd_A": number(row.get("pocket_rms")),
        "pocket_length": number(row.get("pocket_len"), int),
        "pocket_distance_A": number(row.get("pocket_dist")),
        "alignment_matrix": text(row.get("aln_matrix")),
        "radius_of_gyration_distance_A": number(row.get("RoG_distance")),
        "source_member": source_member,
    }


def selected_query_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return values or None


def flush_rows(
    writer: pq.ParquetWriter | None,
    rows: list[dict[str, object]],
    output: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    if writer is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output, SCHEMA, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def ingest(
    *,
    archive_path: Path,
    output_path: Path,
    states: set[str],
    query_filter: set[str] | None,
    max_queries: int,
    batch_rows: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    query_ids: set[str] = set()
    counts: Counter[str] = Counter()
    members_read = 0

    try:
        with tarfile.open(archive_path, mode="r|gz") as outer:
            inner_file = None
            for member in outer:
                if member.name.endswith("/data.tar.gz"):
                    inner_file = outer.extractfile(member)
                    break
            if inner_file is None:
                raise ValueError("AHoJ outer archive has no data.tar.gz member")

            with inner_file, tarfile.open(fileobj=inner_file, mode="r|gz") as inner:
                for member in inner:
                    if not member.isfile():
                        continue
                    path_fields = Path(member.name).parts
                    if len(path_fields) != 4:
                        continue
                    _data, _batch, query_id, filename = path_fields
                    state = RELATION_FILES.get(filename)
                    if state is None or state not in states:
                        continue
                    if query_filter is not None and query_id not in query_filter:
                        continue
                    if query_id not in query_ids:
                        if max_queries and len(query_ids) >= max_queries:
                            break
                        query_ids.add(query_id)

                    extracted = inner.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"cannot read nested member {member.name}")
                    members_read += 1
                    # ``tarfile`` stream-mode members are intentionally
                    # non-seekable and do not expose ``seekable()``; CSV files
                    # are per-query and small enough to decode one at a time.
                    with extracted:
                        member_text = extracted.read().decode("utf-8")
                    with io.StringIO(member_text, newline="") as handle:
                        reader = csv.DictReader(handle)
                        for source_row in reader:
                            # global_results contains all candidates; only its
                            # deposited query-holo row is unique information.
                            if filename == "global_results.csv" and source_row.get(
                                "role"
                            ) != "Q":
                                continue
                            rows.append(
                                relation_row(
                                    source_row,
                                    query_id=query_id,
                                    state=state,
                                    source_member=member.name,
                                )
                            )
                            counts[state] += 1
                            if len(rows) >= batch_rows:
                                writer = flush_rows(writer, rows, output_path)
                            if filename == "global_results.csv":
                                break
    finally:
        if rows:
            writer = flush_rows(writer, rows, output_path)
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("AHoJ ingestion produced no rows")
    return {
        "kind": "voxbind_v4_ahoj_relation_ingest",
        "archive": str(archive_path),
        "output": str(output_path),
        "states": sorted(states),
        "query_filter": query_filter is not None,
        "max_queries": max_queries,
        "n_queries": len(query_ids),
        "n_members_read": members_read,
        "rows": dict(counts),
        "n_rows": sum(counts.values()),
        "schema": [field.name for field in SCHEMA],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=root
        / "dataset"
        / "data"
        / "ahoj"
        / "downloads"
        / "ahojdb_v2c.tar.gz",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=root
        / "dataset"
        / "data"
        / "ahoj"
        / "downloads"
        / "db_stats_v2c.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset" / "data" / "v4" / "ahoj_relations.parquet",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=sorted(set(RELATION_FILES.values())),
        default=sorted(set(RELATION_FILES.values())),
    )
    parser.add_argument(
        "--query-ids",
        type=Path,
        help="optional one-AHoJ-query-ID-per-line filter",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument(
        "--skip-archive-verification",
        action="store_true",
        help="skip the expensive MD5 pass only if ahoj_v2c.json is already frozen",
    )
    args = parser.parse_args()

    if not args.skip_archive_verification:
        verify_archive(args.archive, args.stats)
    report = ingest(
        archive_path=args.archive,
        output_path=args.output,
        states=set(args.states),
        query_filter=selected_query_ids(args.query_ids),
        max_queries=args.max_queries,
        batch_rows=args.batch_rows,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
