"""01d_pdbbind_ligand_rscc.py — fetch per-ligand real-space CC (RSCC) for PDBbind complexes.

For the "LP-PDBBind ∩ ED-available with BOTH protein & ligand density" downstream dataset, we
need a ligand-resolution check analogous to the PLINDER pretraining filter (ligand RSCC ≥ 0.8).
PLINDER shipped RSCC in its annotation table; PDBbind does not, so we pull it from the wwPDB
validation report (RCSB mirror), which carries per-residue RSCC.

Per complex: the ligand = the largest non-standard / non-junk HETATM residue (drug-like, ≥6 atoms);
RSCC = best-resolved copy of that residue (max over copies). Protein-side density is already
guaranteed by has_density (the 2Fo-Fc map covers the pocket), and we additionally report the
pocket-residue median RSCC so callers can require BOTH ≥ threshold.

Concurrent, incremental (resumes by skipping pids already in the out CSV), polite retries.

Run:
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    python dataset/01d_pdbbind_ligand_rscc.py --workers 24 \
        --out dataset/data/pdbbind/ligand_rscc.csv
"""
import argparse, csv, gzip, os, sys, time, urllib.request, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PDB = Path(__file__).resolve().parent / "data" / "pdbbind"
VAL_URL = "https://files.rcsb.org/pub/pdb/validation_reports/{ab}/{pid}/{pid}_validation.xml.gz"

AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO',
      'SER','THR','TRP','TYR','VAL','MSE','SEC','PYL','UNK'}
NUC = {'A','C','G','U','T','DA','DC','DG','DT','DU','DI','N'}
JUNK = {'HOH','DOD','SO4','PO4','GOL','EDO','PEG','PG4','PGE','1PE','ACT','DMS','MPD','TRS','EPE',
        'FMT','NA','CL','MG','ZN','CA','K','MN','FE','FE2','CU','NI','CO','CD','HG','BR','IOD','NO3',
        'CO3','IMD','BME','CIT','TLA','FLC','MES','CAC','NH4','ACE','NAG','BMA','MAN','FUC','GAL','BGC',
        'GLC','XYP','SIA','PO4','UNX','UNL','PLP','SF4','FES','OXY','PER','OH','O','MOH','EOH','BU3',
        'P6G','12P','15P','2PE','ACY','SCN','AZI','PG0','POP','SPM','SPD','BEN'}


def vurl(pid):
    pid = pid.lower()
    return VAL_URL.format(ab=pid[1:3], pid=pid)


def fetch_rscc(pid, retries=2, timeout=30):
    """→ dict(pid, status, lig_resname, lig_rscc, lig_natoms, n_copies, poc_rscc_median, n_poc)."""
    err = ""
    for k in range(retries + 1):
        try:
            raw = urllib.request.urlopen(vurl(pid), timeout=timeout).read()
            xml = gzip.decompress(raw)
            break
        except Exception as e:
            err = f"{type(e).__name__}"
            if k == retries:
                return dict(pid=pid, status=f"fetch_err:{err}", lig_resname="", lig_rscc="",
                            lig_natoms="", n_copies="", poc_rscc_median="", n_poc="")
            time.sleep(1.0 + k)
    try:
        root = ET.fromstring(xml)
    except Exception as e:
        return dict(pid=pid, status=f"parse_err:{type(e).__name__}", lig_resname="", lig_rscc="",
                    lig_natoms="", n_copies="", poc_rscc_median="", n_poc="")
    ligs, poc = {}, []
    for sg in root.iter("ModelledSubgroup"):
        a = sg.attrib
        rscc = a.get("rscc")
        if rscc is None:
            continue
        try:
            rscc = float(rscc)
        except ValueError:
            continue
        rn = a.get("resname")
        if rn in AA or rn in NUC:
            poc.append(rscc)
        elif rn and rn not in JUNK:
            na = a.get("natoms") or a.get("NatomsEDS") or 0
            try:
                na = int(na)
            except ValueError:
                na = 0
            ligs.setdefault(rn, []).append((rscc, na))
    poc_med = (sorted(poc)[len(poc) // 2] if poc else "")
    # ligand = non-junk HETATM resname with the most atoms (>=6), best-resolved copy
    drug = {rn: v for rn, v in ligs.items() if max(n for _, n in v) >= 6}
    if not drug:
        return dict(pid=pid, status="no_ligand", lig_resname="", lig_rscc="", lig_natoms="",
                    n_copies="", poc_rscc_median=poc_med, n_poc=len(poc))
    rn = max(drug, key=lambda r: max(n for _, n in drug[r]))
    copies = drug[rn]
    best = max(copies, key=lambda c: c[0])
    return dict(pid=pid, status="ok", lig_resname=rn, lig_rscc=round(best[0], 4),
                lig_natoms=best[1], n_copies=len(copies), poc_rscc_median=(round(poc_med, 4) if poc_med != "" else ""),
                n_poc=len(poc))


def pool_pids():
    lp = pd.read_csv(PDB / "raw" / "LP_PDBBind.csv")
    lp["pid"] = lp["Unnamed: 0"].astype(str).str.lower()
    av = pd.read_csv(PDB / "voxels_v5" / "availability.csv")
    av["pid"] = av.pdb_id.astype(str).str.lower()
    m = lp.merge(av[["pid", "has_density", "filtered"]], on="pid", how="left")
    pool = m[(m.has_density == True) & (m.filtered != True)]
    return pool.pid.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PDB / "ligand_rscc.csv"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    pids = pool_pids()
    if args.limit:
        pids = pids[:args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out).pid.astype(str))
    todo = [p for p in pids if p not in done]
    print(f"pool={len(pids)} | done={len(done)} | todo={len(todo)} | workers={args.workers}", flush=True)

    cols = ["pid", "status", "lig_resname", "lig_rscc", "lig_natoms", "n_copies", "poc_rscc_median", "n_poc"]
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_rscc, p): p for p in todo}
        for fut in as_completed(futs):
            w.writerow(fut.result()); n += 1
            if n % 200 == 0:
                f.flush(); dt = time.time() - t0
                print(f"  {n}/{len(todo)}  ({n/dt:.1f}/s, ~{(len(todo)-n)/max(n/dt,1e-9)/60:.0f} min left)", flush=True)
    f.close()
    df = pd.read_csv(out)
    ok = df[df.status == "ok"]
    print(f"\n[done] {len(df)} rows | ok={len(ok)} | no_ligand={int((df.status=='no_ligand').sum())} | "
          f"err={int(df.status.str.contains('err').sum())}", flush=True)
    if len(ok):
        print(f"  lig_rscc>=0.8: {int((ok.lig_rscc>=0.8).sum())} | >=0.7: {int((ok.lig_rscc>=0.7).sum())}", flush=True)


if __name__ == "__main__":
    main()
