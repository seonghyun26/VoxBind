"""02d_misato_voxelize.py — storage-aware chunked voxelize + frozen-feature
extraction for the MISATO QM expansion (built structures).

For each chunk of complexes (structure in misato_qm_built/ + map in pdbbind/ccp4/):
  1. 01b voxelize  uniform atoms + density   -> voxels_mqb/{atoms,density}
  2. 01b voxelize  ligvdw atoms (--no_density)-> voxels_mqb_ligvdw/atoms
  3. 01b poolnorm  v5 (CrossDocked ref stats) -> voxels_mqb_v5/density
  4. encode 3 conditions {density_gradmag, atomblob_ligvdw, atomblob_density_gradmag}
     (gradmag derived from the v5 crop), append 512-D vecs to feature accumulators
  5. DELETE the chunk's atom + raw-density voxels (the 11.5 MB/complex hog); keep
     only the tiny v5 density crops + the accumulated features.

Resumable: skips pids already in every accumulator. Run with voxbind python.
Features merge into dataset/data/pdbbind/features/{cond}_e99_v5.pt (existing pool +
new complexes), so the existing `probe --split misato` reads the grown pool unchanged.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import gemmi  # noqa: F401  (load order before torch)
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "data" / "pdbbind"
PYBIN = sys.executable
REF_STATS = HERE.parent / "data" / "xray_crops_aligned_v5" / "stats.json"

# separate dirs so we never touch the pbpp-2020 voxel caches / metadata
MQB        = PB / "voxels_mqb"
MQB_LIGVDW = PB / "voxels_mqb_ligvdw"
MQB_V5     = PB / "voxels_mqb_v5"
JUNK       = PB / "voxels_mqb_junk"          # throwaway v2/v3/v4
CHUNK_CSV  = PB / "_mqb_chunk.csv"
STRUCT_DIR = PB / "structures" / "misato_qm_built"
CCP4_DIR   = PB / "ccp4"
FEAT_DIR   = PB / "features"

CONDS = ["density_gradmag", "atomblob_ligvdw", "atomblob_density_gradmag"]


def load_01c():
    spec = importlib.util.spec_from_file_location("p01c", HERE / "01c_pdbbind_probe.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run01b(args, log):
    cmd = [PYBIN, str(HERE / "01b_pdbbind_preprocess.py")] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.write(f"\n[01b FAIL] {' '.join(args)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
        log.flush()
    return r.returncode == 0


def voxelize_chunk(pids, log):
    import csv
    with open(CHUNK_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pdb_id", "has_struct"])
        for p in pids:
            w.writerow([p, True])
    common = ["--index_csv", str(CHUNK_CSV), "--struct_dir", str(STRUCT_DIR),
              "--element_filter", "ligand", "--device", "cpu", "--allow_stale_cache"]
    ok = run01b(["voxelize", *common, "--ccp4_dir", str(CCP4_DIR), "--out_dir", str(MQB)], log)
    ok &= run01b(["voxelize", *common, "--ligand_vdw", "--no_density",
                  "--out_dir", str(MQB_LIGVDW)], log)
    ok &= run01b(["poolnorm", "--index_csv", str(CHUNK_CSV), "--struct_dir", str(STRUCT_DIR),
                  "--ccp4_dir", str(CCP4_DIR), "--element_filter", "ligand",
                  "--atoms_dir", str(MQB / "atoms"),
                  "--v2_dir", str(JUNK / "v2"), "--v3_dir", str(JUNK / "v3"),
                  "--v4_dir", str(JUNK / "v4"), "--v5_dir", str(MQB_V5),
                  "--v5_stats_source", "reference",
                  "--v5_reference_stats_json", str(REF_STATS)], log)
    return ok


def delete_chunk_atoms(pids):
    for p in pids:
        for f in (MQB / "atoms" / f"{p}.npy", MQB_LIGVDW / "atoms" / f"{p}.npy",
                  MQB / "density" / f"{p}.npy"):
            try: f.unlink()
            except FileNotFoundError: pass
    # clear throwaway v2/v3/v4 crops
    for v in ("v2", "v3", "v4"):
        d = JUNK / v / "density"
        if d.is_dir():
            for p in pids:
                try: (d / f"{p}.npy").unlink()
                except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new pids (0=all)")
    ap.add_argument("--save_every", type=int, default=2, help="save feature .pt every N chunks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for d in (MQB, MQB_LIGVDW, MQB_V5, JUNK):
        d.mkdir(parents=True, exist_ok=True)
    log = open(HERE.parent / "log" / "260612_mqb_driver.log", "a")
    c = load_01c()
    from omegaconf import OmegaConf

    # encoders + specs
    enc, spec = {}, {}
    for cond in CONDS:
        exp = c.resolve_exp(cond, "v5")
        cfg = OmegaConf.load(exp / "cfg.yaml")
        enc[cond] = c.load_encoder(exp, 99, args.device, cfg=cfg)
        spec[cond] = c.infer_feature_spec(cond, cfg, "auto")
    n_in = {cond: spec[cond].expected_channels for cond in CONDS}

    # accumulators: start from existing feature files
    accum, meta = {}, {}
    for cond in CONDS:
        fp = FEAT_DIR / f"{cond}_e99_v5.pt"
        d = torch.load(fp, map_location="cpu", weights_only=False)
        accum[cond] = dict(d["features"]); meta[cond] = d
        print(f"  {cond}: start {len(accum[cond])} features")

    # ready pids = built structure + map present, not already in ALL accumulators
    built = {p.name for p in STRUCT_DIR.iterdir() if p.is_dir()}
    have_map = lambda p: (CCP4_DIR / f"{p}.ccp4").exists()
    done = set.intersection(*[set(accum[c]) for c in CONDS])
    ready = sorted(p for p in built if have_map(p) and p not in done)
    if args.limit:
        ready = ready[:args.limit]
    print(f"  ready to process: {len(ready)} pids (chunk={args.chunk})", flush=True)

    def atom_dir_of(cond):
        return MQB_LIGVDW / "atoms"      # both ligvdw conditions use this
    def save():
        for cond in CONDS:
            out = dict(meta[cond]); out["features"] = accum[cond]
            tmp = FEAT_DIR / f"{cond}_e99_v5.pt.tmp"
            torch.save(out, tmp); os.replace(tmp, FEAT_DIR / f"{cond}_e99_v5.pt")

    for ci in range(0, len(ready), args.chunk):
        chunk = ready[ci:ci + args.chunk]
        voxelize_chunk(chunk, log)
        n_new = 0
        for cond in CONDS:
            ad = atom_dir_of(cond)
            for pid in chunk:
                if pid in accum[cond]:
                    continue
                try:
                    x = c.load_voxels_for(pid, cond, n_in[cond], ad, MQB_V5 / "density",
                                          input_mode=spec[cond].input_mode,
                                          with_gradmag=spec[cond].with_gradmag,
                                          gradmag_dir=None).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        v = c.encode_tokens(enc[cond], x).mean(1).squeeze(0).cpu().contiguous().clone()
                    accum[cond][pid] = v; n_new += 1
                except Exception as e:
                    log.write(f"[feat {cond} {pid}] {repr(e)[:120]}\n"); log.flush()
        delete_chunk_atoms(chunk)
        done_now = len(set.intersection(*[set(accum[x]) for x in CONDS]))
        print(f"  chunk {ci//args.chunk+1}: +{n_new} feats, pool now {done_now}", flush=True)
        if (ci // args.chunk + 1) % args.save_every == 0:
            save()
    save()
    log.close()
    print("done. feature pools:", {cond: len(accum[cond]) for cond in CONDS})


if __name__ == "__main__":
    main()
