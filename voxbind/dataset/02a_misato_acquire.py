"""02a_misato_acquire.py — acquire raw MISATO inputs.

Stage 02a of the MISATO pipeline (02a acquire → 02b targets → 02c structures →
02d voxelize → 01c probe). Two independent fetches, both resumable and gentle
(the Zenodo/PDB-REDO lesson: bursts get the IP throttled/blocked, so low
concurrency + resume, never a flood):

  hdf5  — MISATO QM.hdf5 (~0.3 GB) + MD.hdf5 (~124 GB) from Zenodo record 7711953.
          Single-stream, resumable (curl -C -). MD.hdf5 is large — expect hours.

  pdbe  — for an id list, the PDBe deposited PDB (pocket source) and 2Fo-Fc EDS
          map (.ccp4) per complex, used by 02c/02d to build + voxelize structures
          PDBbind lacks. Skips existing files; logs 404s (non-EDS entries).

Env: none special (shells out to curl — avoids the voxbind urllib proxy quirk).

    python dataset/02a_misato_acquire.py hdf5 --which both
    python dataset/02a_misato_acquire.py pdbe --ids <idlist.txt> --what both
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MISATO = HERE / "data" / "misato"
PB = HERE / "data" / "pdbbind"

ZENODO = "https://zenodo.org/records/7711953/files/{name}?download=1"
MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
PDB_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent"


def curl(url: str, dst: Path, resume: bool = False, timeout: int = 40) -> int:
    """Fetch url→dst with curl. Returns HTTP code (or 0 on transport error)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "--max-time", str(timeout), "--retry", "2", "--retry-delay", "2",
           "-w", "%{http_code}", "-o", str(dst)]
    if resume:
        cmd += ["-C", "-"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(r.stdout.strip()[-3:])
    except ValueError:
        return 0


# ── hdf5 (Zenodo) ────────────────────────────────────────────────────────────
def cmd_hdf5(args):
    names = {"qm": ["QM.hdf5"], "md": ["MD.hdf5"], "both": ["QM.hdf5", "MD.hdf5"]}[args.which]
    MISATO.mkdir(parents=True, exist_ok=True)
    for name in names:
        dst = MISATO / name
        print(f"[hdf5] {name} → {dst}  (resumable; single stream, gentle)")
        # long timeout for the 124 GB MD file; resume on restart
        code = curl(ZENODO.format(name=name), dst, resume=True, timeout=args.timeout)
        ok = dst.exists() and dst.stat().st_size > 0
        print(f"  http={code}  size={dst.stat().st_size/1e9:.1f} GB" if ok else f"  FAILED http={code}")


# ── pdbe (deposited PDB + EDS map) ─────────────────────────────────────────────
def _fetch_pdbe(pid: str, what: str, ccp4_dir: Path, pdb_dir: Path) -> tuple[str, str]:
    out = []
    if what in ("maps", "both"):
        dst = ccp4_dir / f"{pid}.ccp4"
        if not (dst.exists() and dst.stat().st_size > 0):
            out.append(f"map:{curl(MAP_URL.format(pid=pid), dst)}")
    if what in ("pdbs", "both"):
        dst = pdb_dir / f"{pid}.pdb"
        if not (dst.exists() and dst.stat().st_size > 0):
            out.append(f"pdb:{curl(PDB_URL.format(pid=pid), dst)}")
    return pid, ",".join(out)


def cmd_pdbe(args):
    ids = [l.strip().lower() for l in open(args.ids) if l.strip()]
    ccp4_dir, pdb_dir = Path(args.ccp4_dir), Path(args.pdb_dir)
    ccp4_dir.mkdir(parents=True, exist_ok=True); pdb_dir.mkdir(parents=True, exist_ok=True)
    fail = []
    print(f"[pdbe] {len(ids)} ids, what={args.what}, concurrency={args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (pid, res) in enumerate(ex.map(
                lambda p: _fetch_pdbe(p, args.what, ccp4_dir, pdb_dir), ids), 1):
            for tok in res.split(","):
                if tok and not tok.endswith(":200"):
                    fail.append(f"{pid} {tok}")
            if n % 1000 == 0:
                print(f"  {n}/{len(ids)}  non-200: {len(fail)}", flush=True)
    if fail:
        fp = ccp4_dir / ".dl_404.txt"; fp.write_text("\n".join(fail) + "\n")
        print(f"  non-200 (mostly non-EDS 404s): {len(fail)} → {fp}")
    print("[pdbe] done")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ph = sub.add_parser("hdf5", help="download MISATO QM/MD HDF5 from Zenodo")
    ph.add_argument("--which", choices=["qm", "md", "both"], default="both")
    ph.add_argument("--timeout", type=int, default=86400, help="curl --max-time (s); MD.hdf5 is huge")
    ph.set_defaults(func=cmd_hdf5)

    pp = sub.add_parser("pdbe", help="download PDBe EDS maps + deposited PDBs for an id list")
    pp.add_argument("--ids", required=True, help="newline-separated PDB IDs")
    pp.add_argument("--what", choices=["maps", "pdbs", "both"], default="both")
    pp.add_argument("--ccp4_dir", default=str(PB / "ccp4"))
    pp.add_argument("--pdb_dir", default=str(HERE / "data" / "pdb"))
    pp.add_argument("--workers", type=int, default=12, help="moderate; PDBe/EBI is robust")
    pp.set_defaults(func=cmd_pdbe)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
