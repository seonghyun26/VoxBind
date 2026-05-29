"""01da_pdbbind_density_pool_norm.py — Phase 4b: pocket-pool density normalisations.

Produces TWO new density-crop variants from the raw CCP4 maps in
dataset/data/pdbbind/ccp4/, normalised against the pool of all pocket crops
(not per-map, not per-crop):

  v2 — pocket-pool z-score
       x' = (x − μ_pool) / σ_pool
       μ_pool, σ_pool computed over EVERY voxel of EVERY pocket crop concatenated.

  v3 — pocket-pool symmetric max-abs
       x' = x / max_abs_pool
       max_abs_pool = max(|x|) across the same pool. Output strict [−1, +1].

Both schemes:
  - SKIP the per-map z-score that `_load_grid` does in v1 (use raw CCP4 values)
  - SKIP the per-crop ±3σ clip that `normalize_crop` does in v1
  - apply ONE dataset-wide statistic derived from all 4,827 pocket crops

Atom voxels (11, 64, 64, 64) are NOT affected — v2 and v3 reuse the existing
voxels/atoms/ directory via symlinks.

Usage
-----
    cd voxbind
    ~/.conda/envs/voxbind/bin/python dataset/01da_pdbbind_density_pool_norm.py

Outputs
-------
    dataset/data/pdbbind/voxels_v2/density/{pid}.npy   float16 (64, 64, 64)
    dataset/data/pdbbind/voxels_v2/atoms/      → symlink → ../voxels/atoms/
    dataset/data/pdbbind/voxels_v2/stats.json          {μ, σ, n_crops, raw_min, raw_max}

    dataset/data/pdbbind/voxels_v3/density/{pid}.npy   float16 (64, 64, 64)
    dataset/data/pdbbind/voxels_v3/atoms/      → symlink → ../voxels/atoms/
    dataset/data/pdbbind/voxels_v3/stats.json          {max_abs, n_crops, raw_min, raw_max}
"""

import argparse
import json
import os
import sys
from pathlib import Path

# gemmi BEFORE numpy/torch — see feedback_gemmi_torch_import_order memory.
import gemmi

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voxbind.dataset.crossdocked_xray import _crop_density


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR  = Path(__file__).parent / "data" / "pdbbind"
INDEX_CSV    = PDBBIND_DIR / "index.csv"
STRUCT_DIR   = PDBBIND_DIR / "structures" / "pbpp-2020"
CCP4_DIR     = PDBBIND_DIR / "ccp4"
ATOMS_DIR_V1 = PDBBIND_DIR / "voxels" / "atoms"

V2_DIR       = PDBBIND_DIR / "voxels_v2"
V3_DIR       = PDBBIND_DIR / "voxels_v3"

GRID_DIM     = 64
RESOLUTION   = 0.25

# Channel-set of ligand heavy atoms used to compute the ligand COM. Mirrors
# ELEMENTS_HASH_CROSSDOCKED channels 0-6 (C, O, N, S, F, Cl, P).
_LIG_HEAVY_ELEMS = {"C", "O", "N", "S", "F", "Cl", "P"}


# ── CCP4 load — RAW values, no per-map z-score ─────────────────────────────────

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


def parse_ligand_com(sdf_path: Path) -> np.ndarray | None:
    """Heavy-atom centre of mass from a ligand SDF V2000 file."""
    try:
        with sdf_path.open() as f:
            lines = f.readlines()
        n_atoms = int(lines[3][:3])
        coords = []
        for i in range(n_atoms):
            line = lines[4 + i]
            el = line[31:34].strip()
            if el not in _LIG_HEAVY_ELEMS:
                continue
            coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        if not coords:
            return None
        return np.array(coords, dtype=np.float32).mean(axis=0)
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pocket-pool dataset-wide density normalisations (v2: z-score, v3: max-abs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index_csv",  default=str(INDEX_CSV))
    p.add_argument("--struct_dir", default=str(STRUCT_DIR))
    p.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    p.add_argument("--atoms_dir",  default=str(ATOMS_DIR_V1),
                   help="Existing v1 atom voxels to symlink from v2/v3")
    p.add_argument("--v2_dir",     default=str(V2_DIR))
    p.add_argument("--v3_dir",     default=str(V3_DIR))
    p.add_argument("--max_complexes", type=int, default=0,
                   help="Limit to first N (0 = all). Smoke testing.")
    args = p.parse_args()

    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    atoms_dir  = Path(args.atoms_dir).resolve()
    v2_dir     = Path(args.v2_dir)
    v3_dir     = Path(args.v3_dir)

    v2_dens = v2_dir / "density"; v2_dens.mkdir(parents=True, exist_ok=True)
    v3_dens = v3_dir / "density"; v3_dens.mkdir(parents=True, exist_ok=True)

    # Symlink atoms (v2/v3 reuse v1's atom voxels)
    for parent in (v2_dir, v3_dir):
        link = parent / "atoms"
        if not link.exists() and not link.is_symlink():
            os.symlink(atoms_dir, link)
            print(f"  [symlink] {link} → {atoms_dir}")

    print("=== PDBbind v2/v3 density normalisations (pocket dataset-wide) ===")
    print(f"  index_csv : {index_csv}")
    print(f"  ccp4_dir  : {ccp4_dir}")
    print(f"  v2_dir    : {v2_dir}")
    print(f"  v3_dir    : {v3_dir}")

    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)].reset_index(drop=True)

    # Only complexes that have a CCP4 map on disk
    pids = [pid for pid in df["pdb_id"].tolist()
            if (ccp4_dir / f"{pid}.ccp4").exists()]
    if args.max_complexes:
        pids = pids[:args.max_complexes]
    print(f"  complexes : {len(pids):,}")

    # ── Pass 1: crop each pocket from RAW CCP4, accumulate dataset-wide stats ─
    crops: dict[str, np.ndarray] = {}    # pid → float16 (G, G, G)
    sum_v = 0.0
    sum_v2 = 0.0
    n_voxels = 0
    min_v = +np.inf
    max_v = -np.inf

    n_skipped = 0
    skip_log: list[tuple[str, str]] = []
    pbar = tqdm(pids, desc="pass 1: crop + accumulate", unit="cplx")
    for pid in pbar:
        g = load_raw_grid(ccp4_dir / f"{pid}.ccp4")
        if g is None:
            n_skipped += 1; skip_log.append((pid, "load_raw_grid")); continue
        arr, frac_T, nu, nv, nw = g

        lig_com = parse_ligand_com(struct_dir / pid / f"{pid}_ligand.sdf")
        if lig_com is None:
            n_skipped += 1; skip_log.append((pid, "parse_ligand_com")); continue

        try:
            crop = _crop_density(arr, frac_T, nu, nv, nw, lig_com,
                                  G=GRID_DIM, res=RESOLUTION, transform=None)
        except Exception as e:
            n_skipped += 1; skip_log.append((pid, f"crop:{e!r}")); continue

        # Accumulate dataset-wide stats in float64 to avoid precision loss.
        c64 = crop.astype(np.float64)
        sum_v   += c64.sum()
        sum_v2  += (c64 * c64).sum()
        n_voxels += c64.size
        min_v = min(min_v, float(c64.min()))
        max_v = max(max_v, float(c64.max()))

        crops[pid] = crop.astype(np.float16)        # save mem for pass 2

    if n_voxels == 0:
        print("[error] no crops produced; aborting.")
        sys.exit(1)

    mu_pool      = sum_v / n_voxels
    var_pool     = sum_v2 / n_voxels - mu_pool * mu_pool
    sigma_pool   = float(np.sqrt(max(var_pool, 0.0)))
    max_abs_pool = max(abs(min_v), abs(max_v))

    print(f"\n── pocket-pool dataset-wide stats ─────────────────────────────")
    print(f"  n_crops          : {len(crops):,}")
    print(f"  n_voxels (total) : {n_voxels:,}")
    print(f"  raw min / max    : [{min_v:+.4f}, {max_v:+.4f}]")
    print(f"  μ_pool           : {mu_pool:+.6f}")
    print(f"  σ_pool           : {sigma_pool:.6f}")
    print(f"  max_abs_pool     : {max_abs_pool:.6f}")
    print(f"  skipped          : {n_skipped:,}")

    stats_v2 = {
        "scheme":         "pocket-pool z-score",
        "formula":        "x' = (x - mu) / sigma",
        "mu":             mu_pool,
        "sigma":          sigma_pool,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    stats_v3 = {
        "scheme":         "pocket-pool symmetric max-abs",
        "formula":        "x' = x / max_abs",
        "max_abs":        max_abs_pool,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    (v2_dir / "stats.json").write_text(json.dumps(stats_v2, indent=2))
    (v3_dir / "stats.json").write_text(json.dumps(stats_v3, indent=2))

    # ── Pass 2: apply v2 and v3 normalisations, save to disk ──────────────────
    print(f"\n── pass 2: apply normalisations + save ────────────────────────")
    for pid, crop16 in tqdm(crops.items(), desc="pass 2: normalise + save", unit="cplx"):
        c = crop16.astype(np.float32)
        c_v2 = ((c - mu_pool) / sigma_pool).astype(np.float16)
        c_v3 = (c / max_abs_pool).astype(np.float16)
        np.save(str(v2_dens / f"{pid}.npy"), c_v2)
        np.save(str(v3_dens / f"{pid}.npy"), c_v3)

    if skip_log:
        skip_path = PDBBIND_DIR / "voxels_v2_v3_skip_log.txt"
        skip_path.write_text("\n".join(f"{p}\t{m}" for p, m in skip_log) + "\n")
        print(f"  skip log         : {skip_path}")

    print(f"\n  v2 crops written : {len(crops):,} → {v2_dens}")
    print(f"  v3 crops written : {len(crops):,} → {v3_dens}")
    print(f"  stats jsons      : {v2_dir / 'stats.json'} | {v3_dir / 'stats.json'}")
    print("\n  Next:  see dataset/data/pdbbind/README.md for how to consume v2/v3 in Phase 5.")


if __name__ == "__main__":
    main()
