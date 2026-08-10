"""make_data.py — build (pid, seq, smiles, pK) CSVs for the seq+SMILES DTA baselines
(DeepDTA / GraphDTA / MolTrans / PLAPT) on lp_edrscc_v2 + the 2019 temporal holdout.

v2 seq+SMILES+pK come straight from LP_PDBBind.csv (has seq/smiles/value columns).
The 2019 holdout pids are OUTSIDE LP, so seq is parsed from {pid}_protein.pdb (either
structure root) and SMILES from {pid}_ligand.sdf via RDKit (obabel fallback).

Writes base/dta/data/{v2_train,v2_val,v2_test,holdout}.csv  (cols: pid,seq,smiles,pK)
"""
import csv, os, subprocess, sys
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "data"; OUT.mkdir(exist_ok=True)
PDB = REPO / "voxbind/dataset/data/pdbbind"
LP_CSV = PDB / "raw" / "LP_PDBBind.csv"
V2_SPLIT = REPO / "voxbind/splits/lp_edrscc_v2.csv"
HOLD_CSV = REPO / "voxbind/splits/holdout2019_eval.csv"
CASF_CSV = REPO / "voxbind/splits/casf2016_eval.csv"
ROOTS = [PDB / "structures/pbpp-2020", PDB / "structures/misato_qm_built"]

sys.path.insert(0, str(REPO / "base/nesso"))
from build_inputs import AA3TO1, parse_chains   # reuse 3→1 map + chain parser


def seq_from_pdb(pid):
    for r in ROOTS:
        p = r / pid / f"{pid}_protein.pdb"
        if p.exists():
            chains = parse_chains(p)                 # {chain: seq}
            s = "".join(chains.values())
            if s:
                return s
    return None


def smiles_from_sdf(pid):
    for r in ROOTS:
        sdf = r / pid / f"{pid}_ligand.sdf"
        if not sdf.exists():
            continue
        try:
            m = next(Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True))
            if m is not None:
                return Chem.MolToSmiles(m)
        except Exception:
            pass
        try:
            out = subprocess.run(["obabel", str(sdf), "-osmi"], capture_output=True,
                                 text=True, timeout=60).stdout.strip()
            if out:
                return out.split("\t")[0].split()[0]
        except Exception:
            pass
    return None


def build_v2():
    split = {r["pid"].lower(): r["split"] for r in csv.DictReader(open(V2_SPLIT))}
    rows = {"train": [], "val": [], "test": []}
    n_skip = 0
    for r in csv.DictReader(open(LP_CSV)):
        pid = str(r[""]).lower()          # pdb_id is the unnamed first column ("header" col = category)
        sp = split.get(pid)
        if sp not in rows:
            continue
        seq, smi, pk = r.get("seq", ""), r.get("smiles", ""), r.get("value", "")
        if not (seq and smi and pk):
            n_skip += 1; continue
        rows[sp].append((pid, seq, smi, float(pk)))
    for sp, rr in rows.items():
        with open(OUT / f"v2_{sp}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "seq", "smiles", "pK"]); w.writerows(rr)
        print(f"v2_{sp}: {len(rr)}")
    if n_skip:
        print(f"  (skipped {n_skip} v2 rows missing seq/smiles/value)")


def build_holdout():
    rr, fails = [], []
    for r in csv.DictReader(open(HOLD_CSV)):
        pid = r["pid"].lower(); pk = float(r["pK"])
        seq = seq_from_pdb(pid); smi = smiles_from_sdf(pid)
        if seq and smi:
            rr.append((pid, seq, smi, pk))
        else:
            fails.append((pid, f"seq={bool(seq)} smi={bool(smi)}"))
    with open(OUT / "holdout.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "seq", "smiles", "pK"]); w.writerows(rr)
    print(f"holdout: {len(rr)} ok, {len(fails)} fail")
    if fails:
        print("  fails:", fails[:10])


def build_casf():
    """CASF-2016 core (214) — seq/SMILES/pK straight from LP (all 214 present)."""
    lp = {str(r[""]).lower(): r for r in csv.DictReader(open(LP_CSV))}
    rr, fails = [], []
    for r in csv.DictReader(open(CASF_CSV)):
        pid = r["pid"].lower(); pk = float(r["pK"])
        row = lp.get(pid)
        if row and row["seq"] and row["smiles"]:
            rr.append((pid, row["seq"], row["smiles"], pk))
        else:
            fails.append(pid)
    with open(OUT / "casf.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "seq", "smiles", "pK"]); w.writerows(rr)
    print(f"casf: {len(rr)} ok, {len(fails)} fail")


if __name__ == "__main__":
    build_v2()
    build_holdout()
    build_casf()
