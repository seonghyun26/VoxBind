"""build_inputs_holdout.py — generate Nesso YAMLs for the 2019 temporal holdout (699).

Unlike build_inputs.py (pbpp-2020 + LP-csv SMILES), the holdout pids are OUTSIDE
LP-PDBBind, so SMILES is read from the ligand .sdf via RDKit, and protein sequence
from {pid}_protein.pdb in EITHER structure root. Writes _holdout2019/yamls/{pid}.yaml
and regenerates all_pids.txt + shard_0{0..3}.
"""
import os, sys, csv, subprocess
from pathlib import Path
import yaml
from rdkit import Chem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_inputs import parse_chains, build_yaml   # reuse

REPO = HERE.parent.parent
ROOTS = [REPO / "voxbind/dataset/data/pdbbind/structures/pbpp-2020",
         REPO / "voxbind/dataset/data/pdbbind/structures/misato_qm_built"]
HOLD = REPO / "voxbind/splits/holdout2019_eval.csv"
OUT = HERE / "_holdout2019" / "yamls"


def resolve(pid):
    for r in ROOTS:
        p = r / pid / f"{pid}_protein.pdb"
        s = r / pid / f"{pid}_ligand.sdf"
        if p.exists() and s.exists():
            return p, s
    return None, None


def smiles_from_sdf(sdf):
    try:
        m = next(Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True))
        if m is not None:
            return Chem.MolToSmiles(m)
    except Exception:
        pass
    # obabel fallback — robust to misato QM sdf valence quirks that RDKit rejects
    try:
        r = subprocess.run(["obabel", str(sdf), "-osmi"], capture_output=True,
                           text=True, timeout=60)
        smi = r.stdout.strip().split("\t")[0].split()[0] if r.stdout.strip() else ""
        return smi or None
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pids = [r["pid"].lower() for r in csv.DictReader(open(HOLD))]
    ok, fails = [], []
    for pid in pids:
        y = OUT / f"{pid}.yaml"
        if y.exists():
            ok.append(pid); continue
        prot, sdf = resolve(pid)
        if prot is None:
            fails.append((pid, "no_struct")); continue
        smi = smiles_from_sdf(sdf)
        if not smi:
            fails.append((pid, "no_smiles")); continue
        seqs = parse_chains(prot)
        if not seqs:
            fails.append((pid, "no_seq")); continue
        with open(y, "w") as f:
            yaml.safe_dump(build_yaml(pid, smi, seqs), f, sort_keys=False)
        ok.append(pid)
    # regenerate all_pids + shards
    hd = HERE / "_holdout2019"
    (hd / "all_pids.txt").write_text("\n".join(ok) + "\n")
    for i in range(4):
        shard = ok[i::4]
        (hd / f"shard_0{i}").write_text("\n".join(shard) + "\n")
    print(f"Nesso holdout yamls: {len(ok)} ok, {len(fails)} fail")
    if fails:
        print("fails:", fails[:20])


if __name__ == "__main__":
    main()
