"""build_inputs.py — generate per-pid Nesso YAML files for the PDBbind test split.

Usage:
    python build_inputs.py [--split lp_edrscc_v2] [--out_dir _edrscc/yamls]

For each test pid:
  1. Extract per-chain sequences from {pid}_protein.pdb (all chains)
  2. Look up SMILES from LP_PDBBind.csv
  3. Write a Nesso YAML with property: affinity

Falls back to pocket-only chains if protein.pdb has issues.
Skips pids with no SMILES or no structure; logs reasons.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # VoxBind/

STRUCT_BASE = REPO / "voxbind/dataset/data/pdbbind/structures/pbpp-2020"
LP_CSV = REPO / "voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv"
SPLITS_DIR = REPO / "voxbind/splits"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def parse_chains(pdb_path):
    """Return dict {chain_id: sequence_str} from ATOM records, ordered."""
    chains = {}
    chain_order = []
    seen_res = {}  # chain -> set of (resnum, icode)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resnum = line[22:26].strip()
            icode = line[26]
            resname = line[17:20].strip()
            key = (resnum, icode)
            if chain not in seen_res:
                seen_res[chain] = set()
                chains[chain] = []
                chain_order.append(chain)
            if key not in seen_res[chain]:
                seen_res[chain].add(key)
                aa = AA3TO1.get(resname, "X")
                chains[chain].append(aa)
    return {ch: "".join(chains[ch]) for ch in chain_order if chains[ch]}


def parse_pocket_chains(pdb_path):
    """Return set of chain IDs that appear in pocket PDB (ATOM or HETATM)."""
    chains = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chains.add(line[21])
    return chains


def build_yaml(pid, smiles, chain_seqs):
    """Build Nesso YAML dict for a single complex."""
    sequences = []
    for ch, seq in chain_seqs.items():
        sequences.append({
            "protein": {
                "id": ch,
                "sequence": seq,
            }
        })
    # Use chain 'L' for ligand (unlikely to conflict with protein chains)
    lig_id = "L"
    # If 'L' conflicts, use '_'
    if lig_id in chain_seqs:
        lig_id = "Z"
    sequences.append({"ligand": {"id": lig_id, "smiles": smiles}})

    doc = {
        "sequences": sequences,
        "properties": [{"affinity": {"binder": lig_id}}],
    }
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="lp_edrscc_v2")
    parser.add_argument("--out_dir", default="_edrscc/yamls")
    parser.add_argument("--also_casf", action="store_true")
    args = parser.parse_args()

    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LP labels+SMILES (unnamed first column = pid)
    lp = pd.read_csv(LP_CSV, index_col=0)
    lp.index = lp.index.str.lower()
    pk_map = lp["value"].to_dict()
    smi_map = lp["smiles"].to_dict()

    # Load test pids
    split_csv = SPLITS_DIR / f"{args.split}.csv"
    split_df = pd.read_csv(split_csv)
    test_pids = split_df[split_df["split"] == "test"]["pid"].str.lower().tolist()
    print(f"[build_inputs] {len(test_pids)} test pids from {split_csv.name}")

    # Optionally also build for CASF
    casf_pids = []
    if args.also_casf:
        casf_df = pd.read_csv(SPLITS_DIR / "casf2016_eval.csv")
        casf_pids = casf_df["pid"].str.lower().tolist()
        print(f"[build_inputs] also including {len(casf_pids)} CASF pids")

    all_pids = list(dict.fromkeys(test_pids + casf_pids))

    skip_log = {}
    written = 0
    for pid in all_pids:
        out_yaml = out_dir / f"{pid}.yaml"
        if out_yaml.exists():
            written += 1
            continue

        prot_pdb = STRUCT_BASE / pid / f"{pid}_protein.pdb"
        poc_pdb  = STRUCT_BASE / pid / f"{pid}_pocket.pdb"

        if not prot_pdb.exists():
            skip_log[pid] = "no_structure"
            continue

        smiles = smi_map.get(pid)
        if not smiles or (isinstance(smiles, float)):
            skip_log[pid] = "no_smiles"
            continue

        # Parse chains from protein PDB
        try:
            chain_seqs = parse_chains(str(prot_pdb))
        except Exception as e:
            skip_log[pid] = f"parse_error:{e}"
            continue

        if not chain_seqs:
            # Fall back: try pocket PDB chains — shouldn't happen but be safe
            skip_log[pid] = "no_chains"
            continue

        # Filter to only chains that appear in pocket if protein is very large (>5 chains)
        if poc_pdb.exists() and len(chain_seqs) > 5:
            poc_chains = parse_pocket_chains(str(poc_pdb))
            filtered = {ch: seq for ch, seq in chain_seqs.items() if ch in poc_chains}
            if filtered:
                chain_seqs = filtered

        doc = build_yaml(pid, smiles, chain_seqs)
        with open(out_yaml, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
        written += 1

    print(f"[build_inputs] wrote {written} YAMLs, skipped {len(skip_log)}")
    if skip_log:
        print("[build_inputs] skipped pids:")
        for pid, reason in skip_log.items():
            print(f"  {pid}: {reason}")

    # Save skip log
    with open(out_dir / "skip_log.json", "w") as f:
        json.dump(skip_log, f, indent=2)

    print(f"[build_inputs] done → {out_dir}")


if __name__ == "__main__":
    main()
