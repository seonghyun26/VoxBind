#!/usr/bin/env python3
"""Fetch PDBe mFo-DFc DIFFERENCE maps for PDBbind probe complexes.

2Fo-Fc density came from PDBe entry-files {pid}.ccp4 (00a_density_download.py).
The parallel Fo-Fc difference map is served at {pid}_diff.ccp4 — download it into
dataset/data/ccp4_diff/{pid}.ccp4 (SAME dir/name convention as the PLINDER diff
maps, so the poolnorm pass reads it identically). Resumable: skips maps already on
disk. Reads a pid list (one lowercase 4-char PDB id per line).

    python dataset/fetch_pdbbind_diff.py --pids /tmp/pdbbind_diff_to_dl.txt --workers 16
"""
import argparse, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DIFF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}_diff.ccp4"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "ccp4_diff"


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.5,
                  status_forcelist=(500, 502, 503, 504),
                  allowed_methods=("GET",))
    ad = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad); s.mount("http://", ad)
    return s


def fetch(pid: str, sess: requests.Session, out_dir: Path) -> tuple[str, str]:
    dest = out_dir / f"{pid}.ccp4"
    if dest.exists() and dest.stat().st_size > 1024:
        return pid, "skip"
    try:
        r = sess.get(DIFF_URL.format(pid=pid), timeout=90, stream=True)
        if r.status_code == 404:
            return pid, "404"
        r.raise_for_status()
        tmp = dest.with_suffix(".ccp4.part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        tmp.rename(dest)
        return pid, "ok"
    except Exception as e:  # noqa
        return pid, f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True, help="file with one pid per line")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pids = [l.strip().lower() for l in Path(args.pids).read_text().splitlines() if l.strip()]
    print(f"pids: {len(pids)}  out: {out_dir}", flush=True)

    sess = _session()
    counts = {"ok": 0, "skip": 0, "404": 0, "err": 0}
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, p, sess, out_dir): p for p in pids}
        for i, fut in enumerate(as_completed(futs), 1):
            pid, status = fut.result()
            key = "err" if status.startswith("err") else status
            counts[key] = counts.get(key, 0) + 1
            if key in ("404", "err"):
                fails.append(f"{pid}\t{status}")
            if i % 250 == 0 or i == len(pids):
                dt = time.time() - t0
                print(f"[{i}/{len(pids)}] ok={counts['ok']} skip={counts['skip']} "
                      f"404={counts['404']} err={counts['err']}  ({dt:.0f}s)", flush=True)
    if fails:
        (out_dir / "pdbbind_diff_failures.txt").write_text("\n".join(fails) + "\n")
    print(f"DONE ok={counts['ok']} skip={counts['skip']} 404={counts['404']} "
          f"err={counts['err']}  failures→{out_dir}/pdbbind_diff_failures.txt", flush=True)


if __name__ == "__main__":
    main()
