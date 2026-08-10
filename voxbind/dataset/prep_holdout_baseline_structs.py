"""prep_holdout_baseline_structs.py — give the 537 misato-only holdout complexes the two files
the graph/seq baselines need: {pid}_protein.pdb (RCSB-deposited, ATOM records) and {pid}_ligand.mol2
(obabel from the existing misato ligand.sdf). Written IN PLACE into structures/misato_qm_built/{pid}/
so every baseline that resolves from that root picks them up automatically.

CAVEAT: the 167 pbpp-2020 complexes use PDBbind-curated protein+mol2; these 537 use RCSB-deposited
protein (ATOM-only clean) + obabel-protonated mol2 — a mild input-quality difference to note.

Usage: cd voxbind && conda run -n voxbind python dataset/prep_holdout_baseline_structs.py \
         --pids /tmp/misato537.txt --workers 16
"""
import argparse, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIS = HERE / "data" / "pdbbind" / "structures" / "misato_qm_built"
RAW = HERE / "data" / "pdbbind" / "pdb"          # cache of deposited PDBs
RAW.mkdir(parents=True, exist_ok=True)

# keep protein: standard ATOM records + MSE (selenomethionine, common modified residue)
_KEEP_HET = {"MSE"}


def fetch_raw(pid):
    dst = RAW / f"{pid}.pdb"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if not data:
            return None
        dst.write_bytes(data)
        return dst
    except Exception:
        return None


def clean_protein(raw_pdb, out_pdb):
    """Keep protein ATOM records (+ MSE HETATM); drop waters, ligands, ions."""
    lines = []
    for l in raw_pdb.read_text(errors="ignore").splitlines():
        rec = l[:6].strip()
        if rec == "ATOM":
            lines.append(l)
        elif rec == "HETATM" and l[17:20].strip() in _KEEP_HET:
            # relabel MSE as MET-like ATOM so downstream parsers treat it as protein
            lines.append("ATOM  " + l[6:])
        elif rec in ("TER", "END"):
            lines.append(l)
    if not any(x[:4] == "ATOM" for x in lines):
        return False
    out_pdb.write_text("\n".join(lines) + "\n")
    return True


def gen_mol2(sdf, out_mol2):
    try:
        subprocess.run(["obabel", str(sdf), "-O", str(out_mol2)],
                       check=True, capture_output=True, timeout=120)
        return out_mol2.exists() and out_mol2.stat().st_size > 0
    except Exception:
        return False


def prep_one(pid):
    d = MIS / pid
    sdf = d / f"{pid}_ligand.sdf"
    prot = d / f"{pid}_protein.pdb"
    mol2 = d / f"{pid}_ligand.mol2"
    if prot.exists() and mol2.exists():
        return pid, "cached", ""
    raw = fetch_raw(pid)
    if raw is None:
        return pid, "fail", "fetch"
    okp = prot.exists() or clean_protein(raw, prot)
    okm = mol2.exists() or (sdf.exists() and gen_mol2(sdf, mol2))
    if okp and okm:
        return pid, "ok", ""
    return pid, "fail", f"prot={okp} mol2={okm}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    pids = [l.strip().lower() for l in open(a.pids) if l.strip()]
    print(f"prepping {len(pids)} complexes -> {MIS}")
    ok = cached = fail = 0
    fails = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(prep_one, p): p for p in pids}
        for i, f in enumerate(as_completed(futs), 1):
            pid, status, msg = f.result()
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            else:
                fail += 1
                fails.append((pid, msg))
            if i % 50 == 0:
                print(f"  {i}/{len(pids)}  ok={ok} cached={cached} fail={fail}", flush=True)
    print(f"\ndone: ok={ok} cached={cached} fail={fail}")
    if fails:
        print("failures:", fails[:30])
        Path("/tmp/misato_prep_fails.txt").write_text("\n".join(p for p, _ in fails))


if __name__ == "__main__":
    main()
