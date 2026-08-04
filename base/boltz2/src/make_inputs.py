"""make_inputs.py — build Boltz-2 affinity-prediction inputs for the lp_edrscc_v2 test split.

For each test complex we emit one Boltz YAML describing the protein (sequence) + ligand
(SMILES) and request the affinity property. Membership comes from the frozen split
(voxbind/splits/lp_edrscc_v2.csv, split==test); sequence/SMILES/pK from LP_PDBBind.csv.

Boltz-2 affinity constraints (docs): the binder ligand must have <= 128 atoms incl. H,
and >56 heavy atoms is "not recommended". We DROP complexes that exceed the hard 128-atom
limit and FLAG (keep) the 56<heavy<=128 ones, recording everything in a manifest so the
scorer knows the evaluable universe (and what was skipped, loudly — no silent truncation).
"""
import os
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                                  # base/boltz2/
REPO = os.path.dirname(os.path.dirname(BASE))                 # VoxBind/
SPLIT = os.path.join(REPO, "voxbind", "splits", "lp_edrscc_v2.csv")
LP = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")

MAX_TOTAL_ATOMS = 128        # Boltz-2 hard limit (heavy + H)
HEAVY_WARN = 56              # above this is "not recommended"


def atom_counts(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    return m.GetNumAtoms(), Chem.AddHs(m).GetNumAtoms()


CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def yaml_for(pid, seq, smiles, msa_mode):
    # LP_PDBBind joins multiple protein chains with ':'. Boltz needs one protein
    # block per chain with a distinct id; the ligand gets the next free id.
    # msa_mode: "server" -> omit msa key (filled by --use_msa_server);
    #           "empty"  -> single-sequence (msa: empty) per chain.
    chains = [c for c in seq.split(":") if c]
    n = len(chains)
    if n + 1 > len(CHAIN_IDS):
        return None  # absurdly many chains; skip
    lig_id = CHAIN_IDS[n]
    msa_line = "        msa: empty\n" if msa_mode == "empty" else ""
    blocks = ["version: 1\n", "sequences:\n"]
    for i, ch in enumerate(chains):
        blocks.append(
            "  - protein:\n"
            f"      id: {CHAIN_IDS[i]}\n"
            f"      sequence: {ch}\n"
            f"{msa_line}"
        )
    blocks.append(
        "  - ligand:\n"
        f"      id: {lig_id}\n"
        f"      smiles: '{smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        f"      binder: {lig_id}\n"
    )
    return "".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(BASE, "inputs"))
    ap.add_argument("--msa", choices=["server", "empty"], default="server",
                    help="server = use --use_msa_server at predict time; empty = single-seq")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N test pids")
    ap.add_argument("--max_seq_len", type=int, default=2000,
                    help="skip targets longer than this (cost/OOM guard)")
    args = ap.parse_args()

    sp = pd.read_csv(SPLIT); sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    test = sp[sp["split"] == "test"]["pid"].tolist()
    d = lp[lp["pid"].isin(test)].set_index("pid")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    n_written = 0
    for pid in test:
        if args.limit and n_written >= args.limit:
            break
        if pid not in d.index:
            rows.append((pid, np.nan, "no_label", 0, 0)); continue
        r = d.loc[pid]
        seq, smiles, pK = r["seq"], r["smiles"], r["pK"]
        if not isinstance(smiles, str) or not isinstance(seq, str):
            rows.append((pid, pK, "missing_seq_or_smiles", 0, 0)); continue
        ac = atom_counts(smiles)
        if ac is None:
            rows.append((pid, pK, "rdkit_unparseable", 0, 0)); continue
        heavy, total = ac
        if total > MAX_TOTAL_ATOMS:
            rows.append((pid, pK, "ligand_too_big", heavy, total)); continue
        if len(seq.replace(":", "")) > args.max_seq_len:
            rows.append((pid, pK, "seq_too_long", heavy, total)); continue
        y = yaml_for(pid, seq, smiles, args.msa)
        if y is None:
            rows.append((pid, pK, "too_many_chains", heavy, total)); continue
        with open(os.path.join(args.out_dir, f"{pid}.yaml"), "w") as f:
            f.write(y)
        status = "ok_large_ligand" if heavy > HEAVY_WARN else "ok"
        rows.append((pid, pK, status, heavy, total))
        n_written += 1

    man = pd.DataFrame(rows, columns=["pid", "pK", "status", "heavy_atoms", "total_atoms"])
    man.to_csv(os.path.join(BASE, "results", "input_manifest.csv"), index=False)
    print(f"test pids: {len(test)}")
    print(man["status"].value_counts().to_dict())
    print(f"YAMLs written: {n_written}  -> {args.out_dir}")
    print(f"manifest -> {os.path.join(BASE, 'results', 'input_manifest.csv')}")


if __name__ == "__main__":
    main()
