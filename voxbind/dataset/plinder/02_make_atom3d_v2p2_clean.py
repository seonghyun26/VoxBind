#!/usr/bin/env python3
"""Remove PLINDER-v2.2-similar entries from downstream ATOM3D val/test.

PLINDER v2.2 itself is left untouched.  Starting from the official ATOM3D
assignment intersected with ``lp_edrscc_v2``, this writes two downstream splits:

* ID30: remove val/test proteins matching PLINDER v2.2 at >=30% identity;
* ID60: remove val/test proteins matching PLINDER v2.2 at >=60% identity.

Both searches require >=80% coverage of query and target (MMseqs2 cov-mode 0).
The downstream train partitions are unchanged.  A downstream PDB is removed if
any of its LP protein chains matches any polymer entity from a PDB that actually
survives the PLINDER-v2.2 load-time filters (first 112,000 positions, then
in-vocabulary ligand filtering).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


ROOT = Path(__file__).resolve().parents[3]
VOX = ROOT / "voxbind"
SPLITS = VOX / "splits"
DATA = VOX / "dataset" / "data"
WORK = DATA / "pretrain" / "_atom3d_v2p2_clean_mmseqs"
MANIFEST = DATA / "pretrain" / "xray_resample_plinder_v2_perelem" / "train_manifest.npz"
TUPLES = DATA / "pretrain" / "data_train_plinder_v2_perelem.pt"
CIF_DIR = DATA / "cif"
LP_CSV = DATA / "pdbbind" / "raw" / "LP_PDBBind.csv"
MMSEQS = ROOT / "base" / "_tools" / "mmseqs" / "bin" / "mmseqs"

SCHEMES = {
    "id30": {
        "input": SPLITS / "atom3d_lba30_edrscc_v2.csv",
        "output": SPLITS / "atom3d_lba30_edrscc_v2_v22clean.csv",
        "threshold": 0.30,
    },
    "id60": {
        "input": SPLITS / "atom3d_lba60_edrscc_v2.csv",
        "output": SPLITS / "atom3d_lba60_edrscc_v2_v22clean.csv",
        "threshold": 0.60,
    },
}
AUDIT_JSON = SPLITS / "atom3d_lba_v22clean_audit.json"
MATCH_TSV = SPLITS / "atom3d_lba_v22clean_matches.tsv"
AA_RE = re.compile(r"[^A-Z]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [str(x) for x in value] if isinstance(value, list) else [str(value)]


def _header_only(path: Path) -> str:
    lines = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("_atom_site."):
                if lines and lines[-1].strip() == "loop_":
                    lines.pop()
                break
            lines.append(line)
    return "".join(lines)


def parse_protein_entities(task: tuple[str, str]) -> tuple[str, list[tuple[str, str]], str]:
    """Worker result: PDB ID, canonical polypeptide entity sequences, error."""
    pid, path_str = task
    path = Path(path_str)
    if not path.exists():
        return pid, [], "missing_cif"
    try:
        data = MMCIF2Dict(StringIO(_header_only(path)))
        entity_ids = _as_list(data.get("_entity_poly.entity_id"))
        types = _as_list(data.get("_entity_poly.type"))
        seqs = _as_list(data.get("_entity_poly.pdbx_seq_one_letter_code_can"))
        if not seqs:
            seqs = _as_list(data.get("_entity_poly.pdbx_seq_one_letter_code"))
        if not (len(entity_ids) == len(types) == len(seqs)):
            return pid, [], "entity_poly_column_mismatch"
        records = []
        for entity_id, polymer_type, sequence in zip(entity_ids, types, seqs):
            if "polypeptide" not in polymer_type.lower():
                continue
            clean = AA_RE.sub("", sequence.upper())
            if len(clean) >= 20:
                records.append((entity_id, clean))
        return pid, records, "" if records else "no_protein_sequence"
    except Exception as exc:
        return pid, [], f"parse_error:{type(exc).__name__}"


def load_v22_pdb_ids(subset_n: int, n_lig_ch: int) -> tuple[list[str], dict]:
    """Reproduce loader order: shuffle -> val-tail drop -> size -> subset -> OOV."""
    manifest = np.load(MANIFEST)
    manifest_pids = [str(pid).lower() for pid in manifest["pdb_id"]]
    tuples = torch.load(TUPLES, map_location="cpu", weights_only=False)
    random.Random(1234).shuffle(tuples)
    tuples = tuples[: len(tuples) - 100]
    tuples = [(pocket, ligand) for pocket, ligand in tuples if ligand["max_len"] <= 30]
    if len(tuples) != len(manifest_pids):
        raise RuntimeError(
            f"tuple/manifest alignment mismatch: {len(tuples)} != {len(manifest_pids)}"
        )
    if subset_n > len(tuples):
        raise ValueError(f"subset_n={subset_n} > {len(tuples)}")
    before = manifest_pids[:subset_n]
    kept = [
        pid for pid, (_, ligand) in zip(before, tuples[:subset_n])
        if int(ligand["atoms_channel"].max()) < n_lig_ch
    ]
    return kept, {
        "manifest_positions": len(manifest_pids),
        "configured_subset_n": subset_n,
        "v22_kept_positions": len(kept),
        "v22_oov_dropped_positions": subset_n - len(kept),
        "v22_unique_pdb": len(set(kept)),
    }


def downstream_records(path: Path, lp_sequences: dict[str, str], label: str):
    rows = list(csv.DictReader(path.open()))
    records = []
    split_of = {row["pid"].lower(): row["split"] for row in rows}
    for pid, split in sorted(split_of.items()):
        if split not in {"val", "test"}:
            continue
        chains = [AA_RE.sub("", part.upper()) for part in lp_sequences.get(pid, "").split(":")]
        chains = [seq for seq in chains if len(seq) >= 20]
        if not chains:
            raise RuntimeError(f"{label}: no usable downstream protein sequence for {pid}")
        for index, sequence in enumerate(chains):
            records.append((f"{label}|{split}|{pid}|chain{index}", sequence))
    return rows, records


def write_fasta(path: Path, records) -> None:
    with path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def run_search(query: Path, target: Path, output: Path, threshold: float, threads: int):
    tmp = WORK / f"tmp_{int(threshold * 100)}"
    cmd = [
        str(MMSEQS), "easy-search", str(query), str(target), str(output), str(tmp),
        "--min-seq-id", str(threshold), "-c", "0.8", "--cov-mode", "0",
        "--max-seqs", "100000", "--threads", str(threads), "-s", "7.5",
        "--exhaustive-search", "1", "--remove-tmp-files", "1", "-v", "1",
        "--format-output", "query,target,fident,qcov,tcov,alnlen,evalue,bits",
    ]
    print("[mmseqs]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_hits(path: Path, benchmark: str) -> list[dict[str, str]]:
    """Keep one strongest qualifying witness per downstream PDB.

    MMseqs can return thousands of PLINDER homologs for one target.  The split
    decision only needs one witness, so retaining every hit needlessly bloats
    the audit artifact on storage-constrained servers.
    """
    raw_fields = ["query", "target", "fident", "qcov", "tcov", "alnlen", "evalue", "bits"]
    best = {}
    with path.open() as handle:
        for line in handle:
            row = dict(zip(raw_fields, line.rstrip("\n").split("\t")))
            qparts, tparts = row["query"].split("|"), row["target"].split("|")
            row.update({
                "benchmark": benchmark,
                "downstream_split": qparts[1],
                "downstream_pid": qparts[2],
                "downstream_chain": qparts[3],
                "plinder_pid": tparts[0],
                "plinder_entity": tparts[1],
            })
            pid = row["downstream_pid"]
            score = (float(row["bits"]), float(row["fident"]), float(row["qcov"]), float(row["tcov"]))
            previous = best.get(pid)
            if previous is None or score > previous[0]:
                best[pid] = (score, row)
    return [item[1] for item in best.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-n", type=int, default=112000)
    parser.add_argument("--n-lig-ch", type=int, default=7)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-search", action="store_true",
        help="reuse existing WORK/matches_id{30,60}.tsv instead of rerunning MMseqs",
    )
    parser.add_argument(
        "--keep-work", action="store_true",
        help="retain the temporary MMseqs FASTA/results directory (default: remove it)",
    )
    args = parser.parse_args()
    if not MMSEQS.exists():
        raise FileNotFoundError(MMSEQS)
    WORK.mkdir(parents=True, exist_ok=True)

    v22_positions, v22_counts = load_v22_pdb_ids(args.subset_n, args.n_lig_ch)
    v22_pids = sorted(set(v22_positions))
    print(f"[v2.2] {v22_counts}", flush=True)

    lp = pd.read_csv(LP_CSV)
    lp_sequences = dict(zip(
        lp["Unnamed: 0"].astype(str).str.lower(), lp["seq"].fillna("").astype(str),
    ))
    source_rows, query_records = {}, {}
    for label, spec in SCHEMES.items():
        source_rows[label], query_records[label] = downstream_records(spec["input"], lp_sequences, label)
        write_fasta(WORK / f"downstream_{label}.fasta", query_records[label])
        counts = Counter(row["split"] for row in source_rows[label])
        print(f"[query] {label}: counts={dict(counts)}, chains={len(query_records[label])}", flush=True)

    print(f"[cif] parsing {len(v22_pids):,} v2.2 PDBs with {args.workers} workers", flush=True)
    tasks = [(pid, str(CIF_DIR / f"{pid}.cif")) for pid in v22_pids]
    target_records, unsearchable = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (pid, entities, error) in enumerate(pool.map(parse_protein_entities, tasks, chunksize=32), 1):
            if entities:
                target_records.extend((f"{pid}|{entity_id}", seq) for entity_id, seq in entities)
            else:
                unsearchable[pid] = error
            if index % 2000 == 0 or index == len(tasks):
                print(
                    f"[cif] {index:,}/{len(tasks):,}; sequences={len(target_records):,}; "
                    f"unsearchable={len(unsearchable):,}", flush=True,
                )
    target_fasta = WORK / "plinder_v22_polymer_entities.fasta"
    write_fasta(target_fasta, target_records)

    all_hits, summaries = [], {}
    for label, spec in SCHEMES.items():
        result = WORK / f"matches_{label}.tsv"
        if args.reuse_search and result.exists():
            print(f"[mmseqs] reusing {result}", flush=True)
        else:
            run_search(WORK / f"downstream_{label}.fasta", target_fasta, result, spec["threshold"], args.threads)
        hits = read_hits(result, label)
        all_hits.extend(hits)
        remove = {row["downstream_pid"] for row in hits}
        # Same deposited structure is definitely present even if its CIF sequence was unparseable.
        direct_unsearchable = {
            row["pid"].lower() for row in source_rows[label]
            if row["split"] in {"val", "test"} and row["pid"].lower() in unsearchable
        }
        remove |= direct_unsearchable
        kept_rows = [
            row for row in source_rows[label]
            if row["split"] == "train" or row["pid"].lower() not in remove
        ]
        with spec["output"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pid", "split"])
            writer.writeheader()
            writer.writerows(kept_rows)
        before = Counter(row["split"] for row in source_rows[label])
        after = Counter(row["split"] for row in kept_rows)
        removed_by_split = Counter(
            row["split"] for row in source_rows[label]
            if row["split"] in {"val", "test"} and row["pid"].lower() in remove
        )
        summaries[label] = {
            "threshold": spec["threshold"],
            "before": dict(before),
            "after": dict(after),
            "removed": dict(removed_by_split),
            "removed_pdb_ids": sorted(remove),
            "direct_unsearchable_overlap": sorted(direct_unsearchable),
            "output": str(spec["output"].relative_to(ROOT)),
            "output_sha256_bytes": sha256(spec["output"]),
        }
        print(f"[split] {label}: {dict(before)} -> {dict(after)}; removed={dict(removed_by_split)}", flush=True)

    columns = [
        "benchmark", "downstream_split", "downstream_pid", "downstream_chain",
        "plinder_pid", "plinder_entity", "fident", "qcov", "tcov", "alnlen", "evalue", "bits",
    ]
    with MATCH_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(all_hits, key=lambda row: tuple(row[k] for k in columns[:6])))

    audit = {
        "name": "atom3d_lba_edrscc_v2_v22clean",
        "description": "Downstream-only removal of PLINDER-v2.2-similar val/test proteins; PLINDER unchanged.",
        "policy": {
            "id30": "remove ID30 val/test hits at identity>=0.30",
            "id60": "remove ID60 val/test hits at identity>=0.60",
            "coverage": 0.8,
            "cov_mode": 0,
            "search": "MMseqs2 easy-search, exhaustive, sensitivity 7.5",
            "downstream_train": "unchanged",
            "downstream_sequence_unit": "LP protein chain (colon-delimited)",
            "plinder_sequence_unit": "every polymer entity in each actual v2.2 PDB",
        },
        "v22_counts": v22_counts,
        "plinder_entity_sequences": len(target_records),
        "plinder_unsearchable_pdb": unsearchable,
        "schemes": summaries,
        "sources": {
            "manifest_sha256": sha256(MANIFEST),
            "tuples_sha256": sha256(TUPLES),
            "lp_csv_sha256": sha256(LP_CSV),
            "input_split_sha256": {label: sha256(spec["input"]) for label, spec in SCHEMES.items()},
            "matches_sha256": sha256(MATCH_TSV),
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[done] {AUDIT_JSON}", flush=True)
    if not args.keep_work:
        shutil.rmtree(WORK)
        print(f"[cleanup] removed temporary work directory {WORK}", flush=True)


if __name__ == "__main__":
    main()
