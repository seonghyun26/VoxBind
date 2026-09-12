#!/usr/bin/env python3
"""Build PLINDER v2.5 by removing LP-PDBBind CL3-test protein homologs.

Sibling of ``02b_make_v2p4_casf_clean.py`` (which decontaminates against the
214-complex CASF-2016 cohort).  v2.5 decontaminates the *actual* PLINDER-v2
per-element training manifest against the ``lp_edrscc_v2_cl123`` **test** cohort
(733 proteins) -- the same test set the ID60/ID30 affinity-novelty tables score.
Identical policy to v2.4 so the two versions are directly comparable:

* DIAMOND 2.1.8 sensitive protein search;
* sequence identity >= 30%; and
* coverage >= 80% of the shorter sequence (max(query_cov, target_cov)).

If any PLINDER chain qualifies, every ligand observation from that PLINDER PDB is
removed.  The 164 GB canonical density box is positionally identical to v2 and is
hard-linked into the v2.5 resample directory; v2.5 changes only the manifest
availability mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
VOX = ROOT / "voxbind"
DATA = VOX / "dataset/data"
SOURCE_SPLIT = VOX / "splits/plinder/v2"
OUTPUT_SPLIT = VOX / "splits/plinder/v2p5"
SOURCE_RESAMPLE = DATA / "pretrain/xray_resample_plinder_v2_perelem"
OUTPUT_RESAMPLE = DATA / "pretrain/xray_resample_plinder_v2p5_perelem"
SOURCE_DATA_FILE = DATA / "pretrain/data_train_plinder_v2_perelem.pt"
PLINDER_FASTA = ROOT / "base/openbind/leakage/plinder_v2_pretrain.fasta"
LP_CSV = DATA / "pdbbind/raw/LP_PDBBind.csv"
COHORT_SPLIT = "lp_edrscc_v2_cl123"
COHORT_PARTITION = "test"
DEFAULT_DIAMOND = Path(
    "/home/shpark/.conda/envs/minimol/lib/python3.9/site-packages/"
    "gget/bins/Linux/diamond"
)
IDENTITY = 30.0
SHORTER_COVERAGE = 80.0
VAL_N = 100
HIT_COLUMNS = [
    "query", "subject", "pident", "alignment_length", "query_length",
    "subject_length", "query_coverage", "subject_coverage", "evalue",
    "bitscore",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diamond_binary(explicit: str | None) -> Path:
    candidate = Path(explicit) if explicit else Path(shutil.which("diamond") or DEFAULT_DIAMOND)
    if not candidate.is_file():
        raise FileNotFoundError("DIAMOND not found; pass --diamond")
    return candidate


def cohort_pids() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from voxbind.splits import load_split

    split = load_split(COHORT_SPLIT)
    return sorted({str(pid).lower() for pid in split[COHORT_PARTITION]})


def write_cohort_fasta(path: Path, pids: list[str]) -> tuple[int, int]:
    labels = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    labels["pid"] = labels.pid.astype(str).str.lower()
    seq_of = dict(zip(labels.pid, labels.seq.fillna("").astype(str)))
    n_chains, n_complexes = 0, 0
    missing = []
    with path.open("w") as handle:
        for pid in pids:
            raw = seq_of.get(pid, "")
            chains = 0
            for chain_index, raw_sequence in enumerate(raw.split(":")):
                sequence = "".join(r for r in raw_sequence.upper() if "A" <= r <= "Z")
                if not sequence:
                    continue
                handle.write(f">{pid}__chain{chain_index}\n{sequence}\n")
                n_chains += 1
                chains += 1
            if chains:
                n_complexes += 1
            else:
                missing.append(pid)
    if missing:
        raise RuntimeError(f"{len(missing)} CL3-test cohort proteins lack sequences: {missing[:5]}")
    return n_complexes, n_chains


def search(binary: Path, cohort_fasta: Path, output: Path, threads: int) -> pd.DataFrame:
    database = output.with_suffix("")
    subprocess.run(
        [str(binary), "makedb", "--in", str(cohort_fasta), "--db", str(database), "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            str(binary), "blastp", "--query", str(PLINDER_FASTA),
            "--db", str(database.with_suffix(".dmnd")), "--out", str(output),
            "--max-target-seqs", "1000", "--evalue", "100", "--id", str(IDENTITY),
            "--sensitive", "--quiet", "--threads", str(threads), "--outfmt", "6",
            "qseqid", "sseqid", "pident", "length", "qlen", "slen",
            "qcovhsp", "scovhsp", "evalue", "bitscore",
        ],
        check=True,
    )
    hits = pd.read_csv(output, sep="\t", header=None, names=HIT_COLUMNS)
    hits = hits[
        (hits.pident >= IDENTITY)
        & (hits[["query_coverage", "subject_coverage"]].max(axis=1) >= SHORTER_COVERAGE)
    ].copy()
    hits["plinder_pdb"] = hits["query"].str.split("_").str[0].str.lower()
    hits["cohort_pdb"] = hits["subject"].str.split("__").str[0].str.lower()
    return hits


def best_witnesses(hits: pd.DataFrame, source_pdbs: set[str]) -> pd.DataFrame:
    hits = hits[hits.plinder_pdb.isin(source_pdbs)].copy()
    best = (
        hits.sort_values(
            ["plinder_pdb", "pident", "bitscore", "query_coverage", "subject_coverage"],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates("plinder_pdb")
        .rename(columns={"query": "plinder_chain", "subject": "cohort_chain"})
    )
    columns = [
        "plinder_pdb", "plinder_chain", "cohort_pdb", "cohort_chain", "pident",
        "query_coverage", "subject_coverage", "alignment_length", "query_length",
        "subject_length", "evalue", "bitscore",
    ]
    return best[columns].sort_values("plinder_pdb").reset_index(drop=True)


def hardlink_box(source: Path, destination: Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"refusing to replace non-source box: {destination}")
    os.link(source, destination)


def write_loader_view(removed: set[str], audit: dict) -> None:
    OUTPUT_RESAMPLE.mkdir(parents=True, exist_ok=True)
    source_manifest_path = SOURCE_RESAMPLE / "train_manifest.npz"
    with np.load(source_manifest_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    pdb_ids = pd.Series(payload["pdb_id"].astype(str)).str.lower()
    source_ok = payload["ok"].astype(bool)
    keep = source_ok & ~pdb_ids.isin(removed).to_numpy()
    payload["ok"] = keep
    payload["cl3test_id30_keep"] = keep
    np.savez(OUTPUT_RESAMPLE / "train_manifest.npz", **payload)
    np.save(OUTPUT_RESAMPLE / "train_available.npy", keep)

    recipe = json.loads((SOURCE_RESAMPLE / "resample.json").read_text())
    recipe.update({
        "name": "plinder_v2p5_cl3test_id30",
        "source_resample_dir": str(SOURCE_RESAMPLE.resolve()),
        "availability_filter": {
            "cohort": f"{COHORT_SPLIT} {COHORT_PARTITION}",
            "identity_min_percent": IDENTITY,
            "shorter_sequence_coverage_min_percent": SHORTER_COVERAGE,
            "removed_positions": int((source_ok & ~keep).sum()),
            "retained_positions": int(keep.sum()),
        },
        "train_time_note": (
            "PLINDER v2.5 reuses the v2 per-element tuples and density rows. "
            "Only positions with train_manifest.ok=True are exposed."
        ),
    })
    (OUTPUT_RESAMPLE / "resample.json").write_text(json.dumps(recipe, indent=2) + "\n")

    source_box = SOURCE_RESAMPLE / "box116.dat"
    target_box = OUTPUT_RESAMPLE / "box116.dat"
    hardlink_box(source_box, target_box)
    box_meta = json.loads((SOURCE_RESAMPLE / "box116_meta.json").read_text())
    box_meta.update({
        "name": "plinder_v2p5_cl3test_id30",
        "source_box": str(source_box.resolve()),
        "available_rows": int(keep.sum()),
        "note": (
            "Hard-linked, position-identical PLINDER-v2 box. The v2.5 manifest "
            "availability mask excludes lp_edrscc_v2_cl123 test ID30 homologs."
        ),
    })
    (OUTPUT_RESAMPLE / "box116_meta.json").write_text(json.dumps(box_meta, indent=2) + "\n")

    contract = {
        "kind": "plinder_v2p5_cl3test_id30_loader_contract",
        "source_data_file": str(SOURCE_DATA_FILE.resolve()),
        "source_resample_dir": str(SOURCE_RESAMPLE.resolve()),
        "resample_dir": str(OUTPUT_RESAMPLE.resolve()),
        "box_path": str(target_box.resolve()),
        "box_is_hardlink_to_source": os.path.samefile(source_box, target_box),
        "source_manifest_sha256": sha256(source_manifest_path),
        "filtered_manifest_sha256": sha256(OUTPUT_RESAMPLE / "train_manifest.npz"),
        "source_manifest_positions": len(keep),
        "source_available_positions": int(source_ok.sum()),
        "removed_manifest_positions": int((source_ok & ~keep).sum()),
        "retained_manifest_positions": int(keep.sum()),
        "subset_n": int(keep.sum()) - VAL_N,
        "subset_val_n": VAL_N,
        "identity_min_percent": IDENTITY,
        "shorter_sequence_coverage_min_percent": SHORTER_COVERAGE,
        "cohort": f"{COHORT_SPLIT} {COHORT_PARTITION}",
        "removed_unique_pdb": len(removed),
    }
    (OUTPUT_RESAMPLE / "loader_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    audit.update(contract)


def write_frozen_selection(removed: set[str], audit: dict, cohort_fasta_sha: str) -> None:
    OUTPUT_SPLIT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SOURCE_SPLIT / "plinder_selected.csv")
    selected["entry_pdb_id"] = selected.entry_pdb_id.astype(str).str.lower()
    kept = selected[~selected.entry_pdb_id.isin(removed)].copy()
    selection_path = OUTPUT_SPLIT / "plinder_selected.csv"
    kept.to_csv(selection_path, index=False)

    source_inputs = json.loads((SOURCE_SPLIT / "plinder_inputs.json").read_text())
    source_inputs.update({
        "name": "plinder_v2p5",
        "description": (
            "PLINDER v2.5: v2 per-element corpus decontaminated against the "
            "lp_edrscc_v2_cl123 test cohort (733 proteins) at ID30."
        ),
        "source_version": "v2",
        "selection_sha256": sha256(selection_path),
        "n_selected": len(kept),
        "materialized_loader": {
            "source_manifest_positions": audit["source_manifest_positions"],
            "retained_manifest_positions": audit["retained_manifest_positions"],
            "subset_n": audit["subset_n"],
            "subset_val_n": audit["subset_val_n"],
            "source_data_file": str(SOURCE_DATA_FILE.relative_to(ROOT)),
            "resample_dir": str(OUTPUT_RESAMPLE.relative_to(ROOT)),
        },
    })
    source_inputs.setdefault("filters", {})["cl3test_sequence_decontamination"] = {
        "cohort": f"{COHORT_SPLIT} {COHORT_PARTITION} (733 proteins)",
        "identity_min_percent": IDENTITY,
        "shorter_sequence_coverage_min_percent": SHORTER_COVERAGE,
        "policy": "drop every ligand observation from a PDB with any qualifying protein-chain hit",
        "removed_unique_pdb": audit["removed_unique_pdb"],
        "removed_manifest_positions": audit["removed_manifest_positions"],
    }
    source_inputs.setdefault("leakage_holdout", {})["cl3test_cohort"] = {
        "split": COHORT_SPLIT, "partition": COHORT_PARTITION,
        "cohort_fasta_sha256": cohort_fasta_sha,
    }
    source_inputs["leakage_holdout"]["plinder_v2_chain_fasta"] = {
        "path": str(PLINDER_FASTA.relative_to(ROOT)), "sha256": sha256(PLINDER_FASTA)
    }
    source_inputs["note"] = (
        "CL3-test-clean v2.5 reuses v2 tuples and the position-aligned density box; "
        "the v2.5 manifest availability mask is the authoritative load-time filter."
    )
    (OUTPUT_SPLIT / "plinder_inputs.json").write_text(json.dumps(source_inputs, indent=2) + "\n")

    funnel = json.loads((SOURCE_SPLIT / "plinder_funnel.json").read_text())
    funnel["funnel"].append({
        "stage": "exclude lp_edrscc_v2_cl123 test ID30 homologs (shorter coverage >= 80%)",
        "ligand_instances": len(kept),
        "unique_pdb": int(kept.entry_pdb_id.nunique()),
    })
    funnel.update({
        "n_selected": len(kept),
        "n_unique_pdb": int(kept.entry_pdb_id.nunique()),
        "n_chemotypes": int(kept.ligand_ccd_code.nunique()),
    })
    (OUTPUT_SPLIT / "plinder_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diamond")
    parser.add_argument("--threads", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()
    binary = diamond_binary(args.diamond)
    for required in (
        SOURCE_SPLIT / "plinder_selected.csv", SOURCE_RESAMPLE / "train_manifest.npz",
        SOURCE_RESAMPLE / "box116.dat", SOURCE_DATA_FILE, PLINDER_FASTA, LP_CSV,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    with np.load(SOURCE_RESAMPLE / "train_manifest.npz", allow_pickle=False) as manifest:
        source_pids = pd.Series(manifest["pdb_id"].astype(str)).str.lower()
        source_ok = manifest["ok"].astype(bool)
    source_pdbs = set(source_pids[source_ok])

    pids = cohort_pids()
    OUTPUT_SPLIT.mkdir(parents=True, exist_ok=True)
    cohort_fasta_frozen = OUTPUT_SPLIT / "cl3test_cohort.fasta"
    cohort_n, cohort_chains = write_cohort_fasta(cohort_fasta_frozen, pids)
    cohort_fasta_sha = sha256(cohort_fasta_frozen)

    with tempfile.TemporaryDirectory(prefix="plinder_v2p5_") as temp:
        temp_dir = Path(temp)
        hits = search(binary, cohort_fasta_frozen, temp_dir / "hits.tsv", args.threads)
    witnesses = best_witnesses(hits, source_pdbs)
    removed = set(witnesses.plinder_pdb)
    removed_positions = int((source_ok & source_pids.isin(removed).to_numpy()).sum())
    retained_positions = int(source_ok.sum()) - removed_positions

    audit = {
        "name": "plinder_v2p5_cl3test_id30",
        "cohort_split": COHORT_SPLIT,
        "cohort_partition": COHORT_PARTITION,
        "cohort_complexes": cohort_n,
        "cohort_chain_queries": cohort_chains,
        "cohort_fasta_sha256": cohort_fasta_sha,
        "identity_min_percent": IDENTITY,
        "shorter_sequence_coverage_min_percent": SHORTER_COVERAGE,
        "diamond_version": subprocess.check_output([str(binary), "version"], text=True).strip(),
        "source_manifest_positions": int(len(source_pids)),
        "source_available_positions": int(source_ok.sum()),
        "source_unique_pdb": len(source_pdbs),
        "removed_manifest_positions": removed_positions,
        "removed_unique_pdb": len(removed),
        "retained_manifest_positions": retained_positions,
        "retained_unique_pdb": len(source_pdbs - removed),
        "removed_position_pct": round(100 * removed_positions / int(source_ok.sum()), 3),
        "removed_pdb_pct": round(100 * len(removed) / len(source_pdbs), 3),
        "subset_n": retained_positions - VAL_N,
        "subset_val_n": VAL_N,
    }
    witnesses.to_csv(OUTPUT_SPLIT / "cl3test_id30_matches.tsv", sep="\t", index=False)
    (OUTPUT_SPLIT / "cl3test_id30_removed_pdbs.txt").write_text("\n".join(sorted(removed)) + "\n")
    write_loader_view(removed, audit)
    write_frozen_selection(removed, audit, cohort_fasta_sha)
    (OUTPUT_SPLIT / "cl3test_id30_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
