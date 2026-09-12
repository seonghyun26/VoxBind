#!/usr/bin/env python3
"""How much of the PLINDER-v2.2 pretraining corpus is similar to CL3 test?

Flips the direction of ``02_make_atom3d_v2p2_clean.py``: instead of removing
downstream test/val proteins similar to pretraining, this measures how many
PLINDER-v2.2 pretraining PDBs (and positions/crops) are >=30% / >=60% identical
to any ``lp_edrscc_v2_cl123`` TEST protein, i.e. how much the pretraining set
would shrink if we deleted CL3-test-similar structures before SSL pretraining.

Same MMseqs2 settings as the v22clean split (cov 0.8, cov-mode 0, s 7.5,
exhaustive).  Reuses the helpers from the numbered build stage.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Import helpers from the numbered stage file (module name starts with a digit).
spec = importlib.util.spec_from_file_location(
    "v22clean", HERE / "02_make_atom3d_v2p2_clean.py"
)
v22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v22)

# CIF parsing is duplicated here (not imported from v22) so it lives in __main__
# and pickles cleanly for ProcessPoolExecutor workers.
_AA_RE = re.compile(r"[^A-Z]")


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


def _as_list(value) -> list:
    if value is None:
        return []
    return [str(x) for x in value] if isinstance(value, list) else [str(value)]


def parse_protein_entities(task):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

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
            clean = _AA_RE.sub("", sequence.upper())
            if len(clean) >= 20:
                records.append((entity_id, clean))
        return pid, records, "" if records else "no_protein_sequence"
    except Exception as exc:
        return pid, [], f"parse_error:{type(exc).__name__}"

WORK = ROOT / "voxbind/dataset/data/pretrain/_cl3_test_vs_v22_mmseqs"
OUT = ROOT / "voxbind/splits/cl3_test_vs_v22_pretrain_audit.json"
AA_RE = _AA_RE
SUBSET_N = 112000
N_LIG_CH = 7
WORKERS = 8
THREADS = 8


def cl3_test_query_records() -> list[tuple[str, str]]:
    import csv

    test_pids = {
        row["pid"].lower()
        for row in csv.DictReader((ROOT / "voxbind/splits/lp_edrscc_v2_cl123.csv").open())
        if row["split"] == "test"
    }
    lp = pd.read_csv(v22.LP_CSV)
    seqs = dict(zip(lp["Unnamed: 0"].astype(str).str.lower(), lp["seq"].fillna("").astype(str)))
    records = []
    missing = []
    for pid in sorted(test_pids):
        chains = [AA_RE.sub("", part.upper()) for part in seqs.get(pid, "").split(":")]
        chains = [s for s in chains if len(s) >= 20]
        if not chains:
            missing.append(pid)
            continue
        for i, s in enumerate(chains):
            records.append((f"cl3test|{pid}|chain{i}", s))
    return records, sorted(test_pids), missing


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    # 1. Reproduce the exact v2.2 pretraining corpus (per-position PDB list).
    positions, counts = v22.load_v22_pdb_ids(SUBSET_N, N_LIG_CH)  # positions: pdb per kept crop
    pos_per_pdb = Counter(positions)
    unique_pdb = sorted(pos_per_pdb)
    print(f"[v2.2] {counts}", flush=True)

    # 2. Query FASTA = CL3 test protein chains.
    query_records, test_pids, missing = cl3_test_query_records()
    v22.write_fasta(WORK / "cl3_test.fasta", query_records)
    print(f"[query] cl3 test pids={len(test_pids)} usable_chains={len(query_records)} "
          f"missing_seq={len(missing)}", flush=True)

    # 3. Target FASTA = PLINDER v2.2 polymer entity sequences (parse CIF headers).
    tasks = [(pid, str(v22.CIF_DIR / f"{pid}.cif")) for pid in unique_pdb]
    target_records, unsearchable = [], {}
    print(f"[cif] parsing {len(unique_pdb):,} v2.2 PDBs with {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, (pid, entities, err) in enumerate(
            pool.map(parse_protein_entities, tasks, chunksize=32), 1
        ):
            if entities:
                target_records.extend((f"{pid}|{eid}", seq) for eid, seq in entities)
            else:
                unsearchable[pid] = err
            if i % 5000 == 0 or i == len(tasks):
                print(f"[cif] {i:,}/{len(tasks):,} seqs={len(target_records):,} "
                      f"unsearchable={len(unsearchable):,}", flush=True)
    target_fasta = WORK / "plinder_v22_polymer_entities.fasta"
    v22.write_fasta(target_fasta, target_records)

    # 4. One MMseqs search at 30% (covers 60% via post-filter).
    result = WORK / "cl3test_vs_v22.m8"
    v22.run_search(WORK / "cl3_test.fasta", target_fasta, result, 0.30, THREADS)

    # 5. Distinct PLINDER PDBs matched at each threshold (qcov>=0.8, tcov>=0.8).
    matched = {30: set(), 60: set()}
    with result.open() as h:
        for line in h:
            q, t, fident, qcov, tcov, *_ = line.rstrip("\n").split("\t")
            if float(qcov) < 0.8 or float(tcov) < 0.8:
                continue
            plinder_pid = t.split("|")[0]
            fid = float(fident)
            if fid >= 0.30:
                matched[30].add(plinder_pid)
            if fid >= 0.60:
                matched[60].add(plinder_pid)

    summary = {
        "similarity_reference": "lp_edrscc_v2_cl123 test proteins",
        "corpus": "PLINDER v2.2 pretraining (data_train_plinder_v2_perelem, subset 112000, OOV-filtered)",
        "rule": "MMseqs2 fident; qcov>=0.8 and tcov>=0.8; remove PLINDER PDB if any chain hits any CL3 test chain",
        "corpus_counts": counts,
        "cl3_test_pids": len(test_pids),
        "cl3_test_missing_seq": missing,
        "plinder_unsearchable": len(unsearchable),
        "schemes": {},
    }
    total_pos = sum(pos_per_pdb.values())
    total_pdb = len(unique_pdb)
    for thr in (30, 60):
        pdbs = matched[thr] & set(pos_per_pdb)  # restrict to corpus PDBs
        pos_removed = sum(pos_per_pdb[p] for p in pdbs)
        summary["schemes"][f"id{thr}"] = {
            "threshold": thr / 100,
            "pretrain_pdb_matched": len(pdbs),
            "pretrain_pdb_total": total_pdb,
            "pretrain_pdb_pct": round(100 * len(pdbs) / total_pdb, 2),
            "pretrain_positions_removed": pos_removed,
            "pretrain_positions_total": total_pos,
            "pretrain_positions_pct": round(100 * pos_removed / total_pos, 2),
        }
        print(f"[id{thr}] PDBs {len(pdbs)}/{total_pdb} ({100*len(pdbs)/total_pdb:.2f}%); "
              f"positions {pos_removed}/{total_pos} ({100*pos_removed/total_pos:.2f}%)", flush=True)

    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
