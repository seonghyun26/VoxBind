"""00e_v2_pool_norm_crossdocked.py — V2 / V3 density crops for CrossDocked.

Re-crops the X-ray-available CrossDocked subset from RAW CCP4 maps (no per-map
z-score, no per-crop ±3σ clip) and applies one or both of the dataset-wide
pocket-pool normalisations:

    v2 :  x' = (x − μ_pool) / σ_pool                # pool z-score
    v3 :  x' = x / max_abs_pool                      # pool symmetric max-abs → [−1, +1]

Pool stats are computed over every voxel of every successfully-cropped pocket
box across the requested splits.

Pipeline:
    Pass 1 — per-PDB-grouped: load raw CCP4 once per PDB, apply the per-sample
             alignment transform (reused from xray_crops_aligned/{split}_transforms.npz,
             produced by 00d), crop 64³ at the ligand COM, save RAW float16 to
             scratch (the first --versions directory). Accumulate stats in float64.
    Pass 2 — read each raw crop once, compute each requested version, save to
             the matching directory. The scratch version is overwritten in
             place; other versions get fresh files.

Both versions consume the same pool stats (μ_pool, σ_pool, max_abs_pool) so
they're derived from a single pool, not two independent ones.

Usage
-----
    # Generate both v2 and v3 in one pass (default):
    cd voxbind
    ~/.conda/envs/voxbind/bin/python dataset/00e_v2_pool_norm_crossdocked.py

    # Generate only v2:
    ~/.conda/envs/voxbind/bin/python dataset/00e_v2_pool_norm_crossdocked.py --versions v2

Outputs (per requested version)
-------------------------------
    dataset/data/xray_crops_aligned_v2/train/{:06d}.npy     float16 (64,64,64)
    dataset/data/xray_crops_aligned_v2/test/{:06d}.npy      float16
    dataset/data/xray_crops_aligned_v2/{split}_available.npy
    dataset/data/xray_crops_aligned_v2/stats.json           {scheme, mu, sigma, ...}
    (and the same under xray_crops_aligned_v3/ if v3 is requested)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# gemmi BEFORE torch — see feedback_gemmi_torch_import_order memory.
import gemmi

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voxbind.dataset.crossdocked_xray import _crop_density


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data"
CCP4_DIR     = DATA_DIR / "ccp4"
V1_CROPS_DIR = DATA_DIR / "xray_crops_aligned"          # source for {split}_transforms.npz

GRID_DIM     = 64
RESOLUTION   = 0.25


# ── CCP4 raw load (NO per-map z-score) ─────────────────────────────────────────

def load_raw_grid(ccp4_path: Path):
    """Return (arr_raw, frac_M_T, nu, nv, nw) or None — RAW CCP4 values."""
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        if not np.isfinite(arr).all():
            return None
        orth_mat = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth_mat)
        return arr, frac_M.T, nu, nv, nw
    except Exception:
        return None


# ── Sample list (mirrors DatasetCrossDockedXray's load order) ─────────────────

def load_split(data_dir: Path, split: str, max_len: int = 30):
    if split in ("train", "val"):
        data = torch.load(data_dir / "data_train.pt", weights_only=False)
        import random as _random
        _random.Random(1234).shuffle(data)
        val_sz = 100
        data = data[:len(data) - val_sz] if split == "train" else data[len(data) - val_sz:]
    else:
        data = torch.load(data_dir / "data_test.pt", weights_only=False)
    data = [(p, l) for p, l in data if l["max_len"] <= max_len]
    return data


def pdb_id_of(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].lower()


def centroid_of(ligand_: dict) -> np.ndarray:
    mask = ligand_["atoms_channel"] < 7
    return ligand_["coords"][mask].float().mean(dim=0).numpy()


# ── Pass 1: crop raw + accumulate stats (writes raw to scratch_dir/{split}/) ──

def pass1_crop_raw(
    split: str,
    samples: list,
    R_arr: np.ndarray, t_arr: np.ndarray, ok_arr: np.ndarray,
    ccp4_dir: Path,
    scratch_split_dir: Path,
):
    n = len(samples)
    available = np.zeros(n, dtype=bool)

    pdb_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (pocket_, _) in enumerate(samples):
        if not ok_arr[i]:
            continue
        pdb_to_indices[pdb_id_of(pocket_["id"])].append(i)

    n_pdbs       = len(pdb_to_indices)
    n_pdbs_found = sum(1 for pid in pdb_to_indices if (ccp4_dir / f"{pid}.ccp4").exists())
    print(f"  [{split}]  {n:,} samples   ok={int(ok_arr.sum()):,}   "
          f"{n_pdbs:,} unique PDBs   {n_pdbs_found:,} with CCP4")

    sum_v = 0.0; sum_v2 = 0.0; n_voxels = 0
    min_v = +np.inf; max_v = -np.inf

    for pdb, indices in tqdm(pdb_to_indices.items(), desc=f"pass 1 {split}", unit="PDB"):
        ccp4_path = ccp4_dir / f"{pdb}.ccp4"
        if not ccp4_path.exists():
            continue
        g = load_raw_grid(ccp4_path)
        if g is None:
            continue
        arr, frac_T, nu, nv, nw = g

        for i in indices:
            _, ligand_ = samples[i]
            centroid = centroid_of(ligand_)
            R = R_arr[i]; t = t_arr[i]
            try:
                crop = _crop_density(arr, frac_T, nu, nv, nw, centroid,
                                      G=GRID_DIM, res=RESOLUTION, transform=(R, t))
            except Exception:
                continue

            c64 = crop.astype(np.float64)
            sum_v   += c64.sum()
            sum_v2  += (c64 * c64).sum()
            n_voxels += c64.size
            min_v = min(min_v, float(c64.min()))
            max_v = max(max_v, float(c64.max()))

            np.save(str(scratch_split_dir / f"{i:06d}.npy"), crop.astype(np.float16))
            available[i] = True

    return {
        "n_total":     int(n),
        "n_available": int(available.sum()),
        "sum":         float(sum_v),
        "sum_sq":      float(sum_v2),
        "n_voxels":    int(n_voxels),
        "min":         float(min_v),
        "max":         float(max_v),
    }, available


# ── Pass 2: read raw once, write all requested versions ───────────────────────

def pass2_apply_versions(
    scratch_split_dir: Path,
    version_specs: list[tuple[str, Path, callable]],
    available: np.ndarray,
):
    """For each raw crop in scratch, compute each requested version and save."""
    scratch_is_v2_or_v3 = any(scratch_split_dir.resolve() == v[1].resolve()
                              for v in version_specs)
    n_done = 0
    indices = np.where(available)[0]
    for i in tqdm(indices, desc=f"pass 2 {scratch_split_dir.name}", unit="file"):
        raw_path = scratch_split_dir / f"{i:06d}.npy"
        if not raw_path.exists():
            continue
        raw = np.load(raw_path).astype(np.float32)
        for ver, ver_dir, norm_fn in version_specs:
            out_path = ver_dir / scratch_split_dir.name / f"{i:06d}.npy"
            np.save(str(out_path), norm_fn(raw).astype(np.float16))
        n_done += 1
    return n_done


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir",    default=str(DATA_DIR))
    p.add_argument("--ccp4_dir",    default=str(CCP4_DIR))
    p.add_argument("--v1_dir",      default=str(V1_CROPS_DIR),
                   help="v1 dir used as the source of {split}_transforms.npz")
    p.add_argument("--out_root",    default=str(DATA_DIR),
                   help="Parent dir where xray_crops_aligned_v{N}/ will be written")
    p.add_argument("--versions",    nargs="+", default=["v2", "v3"],
                   choices=["v2", "v3"],
                   help="Which pool-normalisations to produce. Defaults to both.")
    p.add_argument("--splits",      nargs="+", default=["train", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--max_len",     type=int, default=30)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    ccp4_dir = Path(args.ccp4_dir)
    v1_dir   = Path(args.v1_dir)
    out_root = Path(args.out_root)

    # One target directory per requested version.
    ver_dirs = {v: out_root / f"xray_crops_aligned_{v}" for v in args.versions}
    for v, d in ver_dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            (d / split).mkdir(parents=True, exist_ok=True)

    # Scratch (where pass 1 writes raw) = the FIRST requested version's dir.
    # In pass 2 the raw files at that location get overwritten in place with
    # that version's normalised crops; any additional versions get fresh files
    # in their own dirs.
    scratch_dir = ver_dirs[args.versions[0]]

    print("=== V2/V3 CrossDocked pool-normalised density crops ===")
    print(f"  data_dir   : {data_dir}")
    print(f"  ccp4_dir   : {ccp4_dir}")
    print(f"  v1_dir     : {v1_dir}  (transforms source)")
    print(f"  versions   : {args.versions}")
    print(f"  splits     : {args.splits}")
    print(f"  scratch    : {scratch_dir}  (raw → in-place norm during pass 2)")
    for v, d in ver_dirs.items():
        print(f"  → {v:<3s}  : {d}")

    # ── Pass 1 ─────────────────────────────────────────────────────────────────
    per_split_stats: dict[str, dict] = {}
    per_split_available: dict[str, np.ndarray] = {}

    for split in args.splits:
        tx_path = v1_dir / f"{split}_transforms.npz"
        if not tx_path.exists():
            print(f"[error] missing {tx_path} — run dataset/00d_align_xray_density.py first.")
            sys.exit(1)
        tx = np.load(tx_path)
        R_arr, t_arr, ok_arr = tx["R"], tx["t"], tx["ok"]

        samples = load_split(data_dir, split, max_len=args.max_len)
        assert len(samples) == len(ok_arr), (
            f"sample-count mismatch for {split}: data={len(samples)}, "
            f"transforms={len(ok_arr)}"
        )

        stats, available = pass1_crop_raw(
            split, samples, R_arr, t_arr, ok_arr, ccp4_dir,
            scratch_dir / split,
        )
        per_split_stats[split] = stats
        per_split_available[split] = available

        # The availability mask is the same for every version (it's about which
        # samples were successfully cropped, not how they were normalised).
        for v_dir in ver_dirs.values():
            np.save(str(v_dir / f"{split}_available.npy"), available)

    # ── Compute pool stats ─────────────────────────────────────────────────────
    sum_v    = sum(s["sum"]      for s in per_split_stats.values())
    sum_v2   = sum(s["sum_sq"]   for s in per_split_stats.values())
    n_voxels = sum(s["n_voxels"] for s in per_split_stats.values())
    min_v    = min(s["min"]      for s in per_split_stats.values())
    max_v    = max(s["max"]      for s in per_split_stats.values())

    if n_voxels == 0:
        print("[error] no voxels accumulated; aborting.")
        sys.exit(1)

    mu_pool      = sum_v / n_voxels
    sigma_pool   = float(np.sqrt(max(sum_v2 / n_voxels - mu_pool * mu_pool, 0.0)))
    max_abs_pool = max(abs(min_v), abs(max_v))

    print(f"\n── pocket-pool stats over {args.splits} ──────────────────────")
    for split, s in per_split_stats.items():
        print(f"  {split:>5}: n_available={s['n_available']:,}  "
              f"n_voxels={s['n_voxels']:,}")
    print(f"  total n_voxels : {n_voxels:,}")
    print(f"  raw min / max  : [{min_v:+.4f}, {max_v:+.4f}]")
    print(f"  μ_pool         : {mu_pool:+.6f}")
    print(f"  σ_pool         : {sigma_pool:.6f}")
    print(f"  max_abs_pool   : {max_abs_pool:.6f}")

    # ── Build version-specific normalisers + stats.json ────────────────────────
    def _v2(raw: np.ndarray) -> np.ndarray:
        return (raw - mu_pool) / sigma_pool

    def _v3(raw: np.ndarray) -> np.ndarray:
        return raw / max_abs_pool

    NORM_FNS = {"v2": _v2, "v3": _v3}
    version_specs: list[tuple[str, Path, callable]] = [
        (v, ver_dirs[v], NORM_FNS[v]) for v in args.versions
    ]

    common_stats = {
        "splits":      args.splits,
        "n_voxels":    n_voxels,
        "raw_min":     min_v,
        "raw_max":     max_v,
        "per_split":   per_split_stats,
    }
    if "v2" in args.versions:
        (ver_dirs["v2"] / "stats.json").write_text(json.dumps({
            "scheme":  "pocket-pool z-score (CrossDocked v2)",
            "formula": "x' = (x - mu) / sigma",
            "mu":      mu_pool,
            "sigma":   sigma_pool,
            **common_stats,
        }, indent=2))
    if "v3" in args.versions:
        (ver_dirs["v3"] / "stats.json").write_text(json.dumps({
            "scheme":  "pocket-pool symmetric max-abs (CrossDocked v3)",
            "formula": "x' = x / max_abs",
            "max_abs": max_abs_pool,
            **common_stats,
        }, indent=2))

    # ── Pass 2 ─────────────────────────────────────────────────────────────────
    print()
    for split in args.splits:
        n_done = pass2_apply_versions(
            scratch_dir / split, version_specs, per_split_available[split],
        )
        print(f"  {split}: wrote {n_done:,} crops across {len(args.versions)} version(s)")

    print(f"\nDone.")
    for v, d in ver_dirs.items():
        print(f"  {v} → {d}")


if __name__ == "__main__":
    main()
