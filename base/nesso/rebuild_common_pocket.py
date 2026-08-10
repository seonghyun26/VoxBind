"""rebuild_common_pocket.py — regenerate Nesso YAMLs for ONLY the 2019 common-set pids, using
the POCKET chain(s) instead of the whole deposited assembly. The full-assembly yamls made Nesso's
ESM embedding time out on large multi-chain proteins (462-aa median → 150s cap); restricting to the
chain(s) that actually contain the binding pocket (from {pid}_pocket.pdb) is both faster and more
correct for affinity. Clears these pids' old predictions so Nesso re-runs them, and writes a shard.

Usage: python rebuild_common_pocket.py --pids /tmp/common346.txt
"""
import argparse, csv, os, subprocess, sys
from pathlib import Path
import yaml
from rdkit import Chem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_inputs import parse_chains, parse_pocket_chains, build_yaml

REPO = HERE.parent.parent
ROOTS = [REPO / "voxbind/dataset/data/pdbbind/structures/pbpp-2020",
         REPO / "voxbind/dataset/data/pdbbind/structures/misato_qm_built"]
OUT = HERE / "_holdout2019" / "yamls"
PREDS = HERE / "_holdout2019" / "outputs" / "predictions"


def resolve(pid):
    for r in ROOTS:
        d = r / pid
        prot, poc, sdf = d / f"{pid}_protein.pdb", d / f"{pid}_pocket.pdb", d / f"{pid}_ligand.sdf"
        if prot.exists() and sdf.exists():
            return prot, (poc if poc.exists() else None), sdf
    return None, None, None


def smiles_from_sdf(sdf):
    try:
        m = next(Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True))
        if m is not None:
            return Chem.MolToSmiles(m)
    except Exception:
        pass
    try:
        r = subprocess.run(["obabel", str(sdf), "-osmi"], capture_output=True, text=True, timeout=60)
        return r.stdout.strip().split("\t")[0].split()[0] if r.stdout.strip() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    a = ap.parse_args()
    pids = [l.strip().lower() for l in open(a.pids) if l.strip()]
    ok, fails = [], []
    for pid in pids:
        prot, poc, sdf = resolve(pid)
        if prot is None:
            fails.append((pid, "no_struct")); continue
        smi = smiles_from_sdf(sdf)
        if not smi:
            fails.append((pid, "no_smiles")); continue
        seqs = parse_chains(prot)
        if poc is not None:                                   # keep only pocket chain(s)
            pc = parse_pocket_chains(poc)
            filt = {c: s for c, s in seqs.items() if c in pc}
            if filt:
                seqs = filt
        if not seqs:
            fails.append((pid, "no_seq")); continue
        with open(OUT / f"{pid}.yaml", "w") as f:             # overwrite full-assembly yaml
            yaml.safe_dump(build_yaml(pid, smi, seqs), f, sort_keys=False)
        # clear old prediction so Nesso recomputes with the pocket-chain input
        af = PREDS / pid / "affinity.json"
        if af.exists():
            af.unlink()
        ok.append(pid)
    # write shards (4-way) for the common pids
    hd = HERE / "_holdout2019"
    for i in range(4):
        (hd / f"shard_common_0{i}").write_text("\n".join(ok[i::4]) + "\n")
    print(f"pocket-chain yamls regenerated: {len(ok)} ok, {len(fails)} fail")
    if fails:
        print("fails:", fails[:15])


if __name__ == "__main__":
    main()
