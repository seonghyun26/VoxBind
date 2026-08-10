"""download_holdout2019.py — fetch ED maps + deposited PDBs for the 2019 holdout pid list.

Reuses 00a_density_download's PDBe fetchers but drives them from an arbitrary pid list
(the 2019 temporal holdout: PDBbind date>=2019, not in our lp_edrscc_v2 train/val) instead
of the CrossDocked splits. Resumable (skips existing non-empty files). Maps that 404 (no EDS)
are recorded, not retried → those pids simply won't be CDG-evaluable.

Usage:  cd voxbind && python dataset/download_holdout2019.py [--pids FILE] [--workers 32]
"""
import argparse, importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("d00a", HERE / "00a_density_download.py")
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)

VOXDATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="/tmp/holdout2019_kdki_pids.txt")
    ap.add_argument("--ccp4_dir", default=str(VOXDATA / "ccp4"))
    ap.add_argument("--pdb_dir", default=str(VOXDATA / "pdb"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    pids = [l.strip().lower() for l in open(a.pids) if l.strip()]
    ccp4, pdb = Path(a.ccp4_dir), Path(a.pdb_dir)
    ccp4.mkdir(parents=True, exist_ok=True); pdb.mkdir(parents=True, exist_ok=True)

    def pending(dest_dir, ext):
        return [p for p in pids if not (dest_dir / f"{p}.{ext}").exists()
                or (dest_dir / f"{p}.{ext}").stat().st_size == 0]

    map_tasks = [(p, D.MAP_URL.format(pid=p), ccp4 / f"{p}.ccp4") for p in pending(ccp4, "ccp4")]
    pdb_tasks = [(p, D.PDB_URL.format(pid=p), pdb / f"{p}.pdb") for p in pending(pdb, "pdb")]
    print(f"2019 holdout: {len(pids)} pids | to fetch: {len(map_tasks)} maps, {len(pdb_tasks)} pdbs", flush=True)

    if map_tasks:
        ok, skip, fail, missing = D.run_download(map_tasks, a.workers, "maps")
        print(f"[maps] ok={ok} fail={fail} (404/no-EDS={len(missing)})", flush=True)
        Path(ccp4 / "_holdout2019_missing_maps.txt").write_text("\n".join(missing))
    if pdb_tasks:
        ok, skip, fail, missing = D.run_download(pdb_tasks, a.workers, "pdbs")
        print(f"[pdbs] ok={ok} fail={fail}", flush=True)

    have_map = sum(1 for p in pids if (ccp4 / f"{p}.ccp4").exists() and (ccp4 / f"{p}.ccp4").stat().st_size)
    print(f"DONE. maps present: {have_map}/{len(pids)}", flush=True)


if __name__ == "__main__":
    main()
