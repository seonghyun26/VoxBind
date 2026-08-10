#!/usr/bin/env python
"""Build LBA-style protein-sequence-identity splits (30% / 60%) on the existing
lp_edrscc_v2 complexes, so we can measure the PROTEIN-novelty axis (Atom3D-LBA
style) on the *same* complexes/features we already probe under LP-PDBBind.

Method (mirrors Atom3D LBA):
  1. Take the union of lp_edrscc_v2 train/val/test pids (same complex pool).
  2. Cluster their protein sequences with mmseqs2 easy-cluster at --min-seq-id
     0.30 and 0.60 (coverage -c 0.8).  Sequences that share >= threshold identity
     land in the same cluster.
  3. Assign WHOLE clusters to train/val/test (seeded, greedy to ~64/14/22 to match
     the lp_edrscc_v2 proportions) so no cluster — hence no similar protein —
     spans splits.  This gives a leak-proof-on-protein split (ligand-agnostic,
     exactly like LBA).

Writes voxbind/splits/lba{30,60}_edrscc.csv  (pid,split), same format as the
committed LP-PDBBind manifests.  Ligand similarity is intentionally NOT controlled
(that is the whole point of the LBA axis).
"""
import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]           # voxbind/
SPLIT_DIR = REPO / "splits"
LP_CSV = REPO / "dataset/data/pdbbind/raw/LP_PDBBind.csv"
BASE_SPLIT = SPLIT_DIR / "lp_edrscc_v2.csv"
MMSEQS = REPO.parent / "base/_tools/mmseqs/bin/mmseqs"


def load_pool():
    rows = list(csv.DictReader(open(BASE_SPLIT)))
    pids = [r["pid"].lower() for r in rows]
    lp = pd.read_csv(LP_CSV)
    lp["pid"] = lp["Unnamed: 0"].astype(str).str.lower()
    seq = dict(zip(lp["pid"], lp["seq"].astype(str)))
    pool = [(p, seq[p]) for p in pids if p in seq and isinstance(seq[p], str) and len(seq[p]) > 0]
    missing = [p for p in pids if p not in seq or not isinstance(seq.get(p), str) or len(seq.get(p, "")) == 0]
    if missing:
        print(f"  [warn] {len(missing)} pids without usable seq (dropped from clustering): {missing[:8]}...")
    return pool


def run_mmseqs(pool, min_seq_id, workdir, threads=2):
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "pool.fasta"
    with open(fasta, "w") as fh:
        for pid, s in pool:
            fh.write(f">{pid}\n{s}\n")
    out_prefix = workdir / f"clu{int(min_seq_id*100)}"
    tmp = workdir / "tmp"
    cmd = [str(MMSEQS), "easy-cluster", str(fasta), str(out_prefix), str(tmp),
           "--min-seq-id", str(min_seq_id), "-c", "0.8", "--cov-mode", "0",
           "--threads", str(threads), "-v", "1"]
    print(f"  running: mmseqs easy-cluster --min-seq-id {min_seq_id} -c 0.8 (threads={threads})")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # cluster tsv: <rep>\t<member>
    clu = defaultdict(list)
    with open(f"{out_prefix}_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            clu[rep].append(mem)
    return clu


def assign(clusters, n_total, seed, frac=(0.643, 0.137, 0.220)):
    """Greedily assign whole clusters to test/val/train to hit target fractions.
    Deterministic: sort clusters by (size desc, rep) then interleave by a seeded
    permutation so large families don't all pile into one split."""
    import random
    reps = sorted(clusters, key=lambda r: (-len(clusters[r]), r))
    rng = random.Random(seed)
    rng.shuffle(reps)
    tgt_test = frac[2] * n_total
    tgt_val = frac[1] * n_total
    split_of = {}
    n_test = n_val = 0
    for r in reps:
        members = clusters[r]
        if n_test < tgt_test:
            s = "test"; n_test += len(members)
        elif n_val < tgt_val:
            s = "val"; n_val += len(members)
        else:
            s = "train"
        for m in members:
            split_of[m] = s
    return split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workdir", default=str(REPO / "dataset/data/pdbbind/_lba_cluster"))
    args = ap.parse_args()

    if not MMSEQS.exists():
        sys.exit(f"mmseqs not found at {MMSEQS}")
    pool = load_pool()
    print(f"pool: {len(pool)} complexes with usable protein seq")
    workdir = Path(args.workdir)

    for tag, msi in (("lba30", 0.30), ("lba60", 0.60)):
        clusters = run_mmseqs(pool, msi, workdir / tag, threads=args.threads)
        n = sum(len(v) for v in clusters.values())
        split_of = assign(clusters, n, args.seed)
        out = SPLIT_DIR / f"{tag}_edrscc.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["pid", "split"])
            for pid, _ in pool:
                w.writerow([pid, split_of[pid]])
        from collections import Counter
        cc = Counter(split_of.values())
        print(f"  {tag}: {len(clusters)} clusters -> "
              f"train={cc['train']} val={cc['val']} test={cc['test']}  -> {out.name}")
    print("done.")


if __name__ == "__main__":
    main()
