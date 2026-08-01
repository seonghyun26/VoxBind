"""Verify and provenance-stamp the official AHoJ-DB v2c archive.

The archive is deliberately retained compressed. It contains a nested
``data.tar.gz`` member, so wholesale extraction would temporarily duplicate
tens of gigabytes before any canonicalization or structure-factor filtering.
Downstream ingestion must stream the nested archive and materialize only compact
metadata tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

AHOJ_VERSION = "v2c"
AHOJ_RELEASE_DATE = "2025-04-03"
AHOJ_URL = "https://apoholo.cz/api/db/archive/download/ahojdb_v2c.tar.gz"
AHOJ_ARCHIVE_BYTES = 40_181_598_291
# The server ETag is Starlette's MD5("<mtime>-<size>"), not a content hash:
# MD5("1743667694.0-40181598291") == d08d... . Keep it as transport
# provenance, but use the separately computed full-file MD5 for reproducibility.
AHOJ_SERVER_ETAG = "d08d439a6a1b8f34770acbdd1f6f8379"
AHOJ_ARCHIVE_MD5 = "f803f4eacde1b31e05eb4c06f352c8aa"
AHOJ_INNER_ARCHIVE_BYTES = 40_611_424_582
AHOJ_STATS = {
    "num_entries": 515_463,
    "num_unique_pdb_ids": 121_029,
    "num_unique_uniprot_ids": 29_463,
    "num_unique_target_ligands": 37_620,
    "num_found_apo_sites": 14_874_483,
    "num_found_holo_sites": 42_925_658,
    "num_found_unobserved_sites": 354_495,
}


def file_md5(path: Path) -> str:
    """Compute the archive's transport checksum."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_outer_header(path: Path) -> dict[str, int]:
    """Read only the leading outer-tar members and nested archive size."""
    members: dict[str, int] = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            members[member.name] = int(member.size)
            if member.name.endswith("/data.tar.gz"):
                break
    return members


def verify_archive(path: Path, stats_path: Path | None = None) -> dict[str, object]:
    """Fail unless local bytes, MD5, nested member, and official stats match."""
    observed_bytes = path.stat().st_size
    if observed_bytes != AHOJ_ARCHIVE_BYTES:
        raise ValueError(
            f"AHoJ archive size {observed_bytes:,} != {AHOJ_ARCHIVE_BYTES:,}"
        )
    observed_md5 = file_md5(path)
    if observed_md5 != AHOJ_ARCHIVE_MD5:
        raise ValueError(
            f"AHoJ archive MD5 {observed_md5} != {AHOJ_ARCHIVE_MD5}"
        )
    members = inspect_outer_header(path)
    inner = next(
        (
            size
            for name, size in members.items()
            if name.endswith("/data.tar.gz")
        ),
        None,
    )
    if inner != AHOJ_INNER_ARCHIVE_BYTES:
        raise ValueError(
            f"AHoJ nested data.tar.gz size {inner} != {AHOJ_INNER_ARCHIVE_BYTES}"
        )

    observed_stats = None
    if stats_path is not None:
        observed_stats = json.loads(stats_path.read_text())
        if observed_stats != AHOJ_STATS:
            raise ValueError(
                f"AHoJ stats mismatch: observed={observed_stats}, expected={AHOJ_STATS}"
            )
    return {
        "kind": "voxbind_v4_ahoj_archive",
        "version": AHOJ_VERSION,
        "release_date": AHOJ_RELEASE_DATE,
        "url": AHOJ_URL,
        "archive_bytes": observed_bytes,
        "archive_md5": observed_md5,
        "server_etag": AHOJ_SERVER_ETAG,
        "server_etag_kind": 'MD5("mtime-size"), not file-content MD5',
        "inner_archive_bytes": inner,
        "stats": observed_stats or AHOJ_STATS,
        "extraction_policy": "keep compressed; stream nested CSV members",
        "verified": True,
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
        default=root / "splits" / "v4" / "ahoj_v2c.json",
    )
    args = parser.parse_args()
    report = verify_archive(args.archive, args.stats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
